import os

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


def create_trainer(dir_path, params):
    os.makedirs(dir_path, exist_ok=True)

    return Trainer(
        **params,
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
                filename="gan-{epoch:02d}-{val_loss_generator:.4f}",
            ),
        ],
        check_val_every_n_epoch=1,
    )
