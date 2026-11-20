"""Single-worker job queue with an on-disk LRU cache of finished results.

Each job works in `cache_dir/<key>/`; `result.json` is written last and marks the entry as done.
A request whose key is done or in flight gets that job back; failed jobs are not cached.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)
ACTIVE = ("queued", "running")
MAX_FINISHED_JOBS = 200  # finished jobs kept in memory (done ones can be reloaded from disk)

Progress = Callable[[str, float], None]


class Busy(Exception):
    """The request is refused for load reasons (per-IP limit or full queue); the message is for the user."""


class JobTimeout(Exception):
    pass


@dataclass
class Job:
    id: str
    key: str
    ip: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = "queued"
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    finished: threading.Event = field(default_factory=threading.Event, repr=False)


class JobQueue:
    """`run(payload, out_dir, progress) -> result dict (JSON-serialisable)` is executed one job at a time.
    `progress(message, fraction)` updates the job and is where the timeout is enforced (between stages)."""

    def __init__(self, run: Callable, max_queue: int, cache_dir: Path, cache_size: int, timeout_s: float = 1200.0):
        self.run, self.max_queue, self.cache_size, self.timeout_s = run, max_queue, cache_size, timeout_s
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="job")

    # ------------------------------------------------------------------ lookup
    def find(self, key: str) -> Job | None:
        """The done or in-flight job for `key` (memory first, then the disk cache), else None."""
        with self._lock:
            job = self._by_key.get(key)
            if job is not None and job.status != "error":
                self._touch(key)
                return job
            result = self.cache_dir / key / "result.json"
            if result.exists():
                try:
                    data = json.loads(result.read_text())
                except ValueError:
                    shutil.rmtree(result.parent, ignore_errors=True)
                    return None
                job = Job(uuid.uuid4().hex, key, "", "done", 1.0, "done", data)
                job.finished.set()
                self._register(job)
                self._touch(key)
                return job
            return None

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def position(self, job_id: str) -> int:
        """0 for the running job, 1 for the next queued job, and so on; 0 for finished jobs."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return 0
            queued = sorted((j for j in self._jobs.values() if j.status == "queued"), key=lambda j: j.created)
            return queued.index(job) + 1

    def active(self) -> int:
        with self._lock:
            return sum(j.status in ACTIVE for j in self._jobs.values())

    def path(self, job: Job) -> Path:
        return self.cache_dir / job.key

    # ------------------------------------------------------------------ submit
    def submit(self, key: str, ip: str, payload) -> Job:
        with self._lock:
            job = self.find(key)
            if job is not None:
                return job
            active = [j for j in self._jobs.values() if j.status in ACTIVE]
            if any(j.ip == ip for j in active):
                raise Busy("you already have a request in progress; wait for it to finish")
            if len(active) >= self.max_queue:
                raise Busy(f"the queue is full ({self.max_queue} requests); try again in a few minutes")
            job = Job(uuid.uuid4().hex, key, ip)
            self._register(job)
            self._pool.submit(self._work, job, payload)
            return job

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ internals
    def _register(self, job: Job) -> None:
        self._jobs[job.id] = job
        self._by_key[job.key] = job

    def _touch(self, key: str) -> None:
        try:
            (self.cache_dir / key / "result.json").touch()
        except FileNotFoundError:
            pass  # still running

    def _work(self, job: Job, payload) -> None:
        out = self.cache_dir / job.key
        deadline = time.monotonic() + self.timeout_s

        def progress(message: str, fraction: float) -> None:
            if time.monotonic() > deadline:
                raise JobTimeout(f"timed out after {self.timeout_s:g} s")
            job.message, job.progress = message, float(fraction)

        job.status, job.message = "running", "starting"
        try:
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            result = self.run(payload, out, progress)
            tmp = out / "result.json.tmp"
            tmp.write_text(json.dumps(result))
            tmp.rename(out / "result.json")
            self._evict()
            job.result, job.progress, job.message, job.status = result, 1.0, "done", "done"
        except Exception as e:  # noqa: BLE001 - the reason is shown to the user
            if not isinstance(e, (JobTimeout, ValueError)):
                log.exception("job %s failed", job.id)
            shutil.rmtree(out, ignore_errors=True)
            job.error, job.status = str(e) or type(e).__name__, "error"
        finally:
            job.finished.set()
            self._prune()

    def _evict(self) -> None:
        """Delete the least recently used finished results beyond `cache_size`."""
        with self._lock:
            done = sorted((p for p in self.cache_dir.iterdir() if (p / "result.json").exists()),
                          key=lambda p: (p / "result.json").stat().st_mtime)
            for p in done[:max(0, len(done) - self.cache_size)]:
                shutil.rmtree(p, ignore_errors=True)
                job = self._by_key.pop(p.name, None)
                if job is not None:
                    self._jobs.pop(job.id, None)

    def _prune(self) -> None:
        with self._lock:
            finished = sorted((j for j in self._jobs.values() if j.status in ("done", "error")), key=lambda j: j.created)
            for j in finished[:max(0, len(finished) - MAX_FINISHED_JOBS)]:
                self._jobs.pop(j.id, None)
                if self._by_key.get(j.key) is j:
                    del self._by_key[j.key]
