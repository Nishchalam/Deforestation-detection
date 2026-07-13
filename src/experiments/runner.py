"""
runner.py

Coordinates model creation, dataset loading, and callbacks execution.
"""

import time
import yaml
from pathlib import Path
import torch
import torch.optim as optim

from src.data.dataloader import create_dataloaders
from src.experiments.experiment import Experiment
from src.experiments.registry import create_model
from src.experiments.tracker import ExperimentTracker
from src.experiments.utils import get_next_experiment_dir, load_yaml_config
from src.training.trainer import Trainer
from src.training.logger import ExperimentLogger
from src.training.utils import set_seed, count_parameters
from src.training.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor, 
    TensorBoardCallback, 
    CSVLogger
)
from src.utils.paths import EXPERIMENTS_DIR

class ExperimentRunner:
    """
    Orchestrates the lifecycle of deep learning experiments.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = load_yaml_config(self.config_path)
        
    def run(self) -> Experiment:
        # 1. Setup seed and device
        set_seed(self.config["experiment"].get("seed", 42))
        device_str = self.config["experiment"].get("device", "cuda")
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)

        # 2. Get next experiment folder
        model_name = self.config["model"]["name"]
        exp_dir = get_next_experiment_dir(model_name, EXPERIMENTS_DIR)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 3. Setup Logger
        logger = ExperimentLogger(exp_dir)
        logger.log_info(f"Loaded config from {self.config_path}")
        logger.log_info(f"Experiment Directory: {exp_dir}")

        # Write config to experiment folder
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f)

        # 4. Instantiate Model via Zoo / Registry
        model_cfg = self.config["model"]
        model = create_model(
            model_cfg["name"],
            in_channels=model_cfg.get("in_channels", 3),
            num_classes=model_cfg.get("num_classes", 10)
        ).to(device)

        logger.log_info(f"Initialized Model: {model_cfg['name']}")
        logger.log_info(f"Trainable Parameters: {count_parameters(model):,}")

        # 5. Dataloaders
        data_cfg = self.config["dataset"]
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=data_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 4)
        )

        # 6. Optimizer & Scheduler
        train_cfg = self.config["training"]
        opt_cfg = train_cfg.get("optimizer", {})
        if opt_cfg.get("name", "adam") == "adam":
            optimizer = optim.Adam(
                model.parameters(), 
                lr=opt_cfg.get("lr", 1e-3),
                weight_decay=opt_cfg.get("weight_decay", 0.0)
            )
        else:
            optimizer = optim.SGD(
                model.parameters(), 
                lr=opt_cfg.get("lr", 1e-3), 
                momentum=0.9
            )
            
        scheduler = None
        sched_cfg = train_cfg.get("scheduler")
        if sched_cfg and sched_cfg.get("name") == "reduce_lr_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=sched_cfg.get("mode", "min"),
                factor=sched_cfg.get("factor", 0.1),
                patience=sched_cfg.get("patience", 3)
            )

        # 7. Callbacks
        callbacks_list = []
        cb_cfg = self.config.get("callbacks", {})
        
        if cb_cfg.get("early_stopping", {}).get("enabled", False):
            es = cb_cfg["early_stopping"]
            callbacks_list.append(EarlyStopping(
                monitor=es.get("monitor", "val_loss"),
                patience=es.get("patience", 5),
                mode=es.get("mode", "min")
            ))
            
        if cb_cfg.get("model_checkpoint", {}).get("enabled", False):
            mc = cb_cfg["model_checkpoint"]
            callbacks_list.append(ModelCheckpoint(
                filepath=exp_dir,
                monitor=mc.get("monitor", "val_accuracy"),
                mode=mc.get("mode", "max")
            ))
            
        if cb_cfg.get("tensorboard", {}).get("enabled", False):
            callbacks_list.append(TensorBoardCallback(logger))
            callbacks_list.append(LearningRateMonitor(logger))
            
        if cb_cfg.get("csv_logger", {}).get("enabled", False):
            callbacks_list.append(CSVLogger(filepath=exp_dir / "history.csv"))

        # Log hyperparams
        logger.log_hyperparams({
            "model": model_cfg["name"],
            "epochs": train_cfg.get("epochs", 20),
            "batch_size": data_cfg.get("batch_size", 32),
            "learning_rate": opt_cfg.get("lr", 1e-3)
        })

        # Create Experiment object
        exp = Experiment(
            name=exp_dir.name,
            config=self.config,
            model=model,
            checkpoint_dir=exp_dir,
            tensorboard_dir=exp_dir / "tensorboard"
        )
        
        # 8. Train
        exp.start()
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=train_cfg.get("epochs", 20),
            device=device,
            loss_name=train_cfg.get("loss", "cross_entropy"),
            scheduler=scheduler,
            callbacks=callbacks_list,
            logger=logger
        )
        trainer.fit()
        exp.end()

        # Gather final metrics
        best_accuracy = 0.0
        for cb in callbacks_list:
            if isinstance(cb, ModelCheckpoint):
                best_accuracy = float(cb.best) if cb.mode == "max" else 0.0
        
        if best_accuracy == 0.0:
            for cb in callbacks_list:
                if isinstance(cb, CSVLogger):
                    val_accs = cb.history.get("val_accuracy", [0.0])
                    best_accuracy = float(max(val_accs))

        # 9. Track and save metadata
        tracker = ExperimentTracker(exp_dir)
        tracker.track(
            name=exp.name,
            training_time=exp.get_duration(),
            best_accuracy=best_accuracy,
            learning_rate=opt_cfg.get("lr", 1e-3),
            optimizer=opt_cfg.get("name", "adam"),
            dataset=data_cfg.get("name", "EuroSAT"),
            checkpoint_path=str(exp_dir / "best_model.pth"),
            tensorboard_path=str(exp_dir / "tensorboard")
        )
        tracker.save()
        
        # Store metadata and history back in Experiment
        exp.history = next((cb.history for cb in callbacks_list if isinstance(cb, CSVLogger)), {})
        exp.metrics = tracker.metadata

        # Create a figures directory
        (exp_dir / "figures").mkdir(exist_ok=True)

        logger.close()
        return exp
