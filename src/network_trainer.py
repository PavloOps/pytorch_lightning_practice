import os
import re

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


def create_trainer(dir_path, params):
    os.makedirs(dir_path, exist_ok=True)
    csv_logger = CSVLogger(save_dir="lightning_logs", name="MyConvNet")

    return Trainer(
        **params,
        logger=csv_logger,
        callbacks=[
            EarlyStopping(monitor="valid/loss", patience=3, mode="min", verbose=False),
            ModelCheckpoint(
                monitor="valid/loss",
                mode="min",
                dirpath=dir_path,
                enable_version_counter=True,
                save_top_k=3,
                auto_insert_metric_name=True,
            ),
        ]
    )


def pick_best_model(dir_path):
    def parse_numbers(filename):
        return tuple(
            int(m.group(1)) if m else 0
            for m in (
                re.search(r"epoch=(\d+)", filename),
                re.search(r"step=(\d+)", filename),
                re.search(r"v(\d+)", filename),
            )
        )

    return max(
        (
            os.path.join(root, f)
            for root, _, files in os.walk(dir_path)
            for f in files
            if f.endswith(".ckpt")
        ),
        key=lambda f: parse_numbers(os.path.basename(f)),
        default=None,
    )
