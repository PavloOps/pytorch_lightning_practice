import os
from dataclasses import asdict

from lightning import Trainer
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)

from .config import CFG


def create_trainer(dir_path, config: CFG, fast_dev_run=False):
    os.makedirs(dir_path, exist_ok=True)
    trainer_params_dict = asdict(config.trainer)

    return Trainer(
        **trainer_params_dict,
        fast_dev_run=fast_dev_run,
        callbacks=[
            EarlyStopping(
                monitor="val/loss_generator",
                patience=5,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                monitor="val/loss_generator",
                mode="min",
                dirpath=dir_path,
                save_top_k=3,
                filename="gan-{epoch:02d}-{val_loss:.4f}",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
