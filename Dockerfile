# CPU-only image for the public explorer (Hugging Face Spaces, Docker SDK). See docs/deploy.md.
# Python 3.12 matches the environment the pinned requirements-app.txt was frozen from.
FROM python:3.12-slim

# libexpat1: the rasterio wheel links it and the slim image does not have it. `ldd` over every .so of the
# installed requirements-app.txt shows no other system library beyond libc, libstdc++, libm and zlib.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/home/user/models \
    CACHE_DIR=/home/user/cache \
    PORT=7860

WORKDIR /home/user/app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# the deforest package (pyproject.toml has no dependencies, they are in requirements-app.txt)
COPY pyproject.toml LICENSE ./
COPY src src
RUN pip install --no-deps .

COPY app app
COPY configs configs
COPY scripts/fetch_weights.py scripts/fetch_weights.py
RUN mkdir -p "$MODEL_DIR" "$CACHE_DIR" && chown -R user:user /home/user

USER user
EXPOSE 7860
ENTRYPOINT ["bash", "app/entrypoint.sh"]
