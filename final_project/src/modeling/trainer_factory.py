from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from lightning import Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import CSVLogger

from final_project.src.config import CFG
from final_project.src.modeling.callbacks import ClearMLSummaryCallback


def create_trainer(
    dir_path: str | Path,
    config: CFG,
    fast_dev_run: bool = False,
    clearml_task: Any = None,
) -> Trainer:
    output_dir = Path(dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer_params = asdict(config.trainer)
    trainer_params["default_root_dir"] = str(output_dir)

    callbacks = _build_callbacks(
        output_dir=output_dir,
        config=config,
        clearml_task=clearml_task,
        fast_dev_run=fast_dev_run,
    )
    logger = _build_logger(output_dir=output_dir)

    return Trainer(
        **trainer_params,
        fast_dev_run=fast_dev_run,
        callbacks=callbacks,
        logger=logger,
    )


def _build_callbacks(
    output_dir: Path,
    config: CFG,
    clearml_task: Any = None,
    fast_dev_run: bool = False,
) -> list[Callback]:
    callbacks: list[Callback] = [
        EarlyStopping(
            monitor="valid/loss",
            patience=config.callbacks.early_stopping_patience,
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="valid/loss",
            mode="min",
            dirpath=str(output_dir),
            save_top_k=config.callbacks.save_top_k,
            save_last=True,
            filename="two-tower-epoch{epoch:02d}-step{step:06d}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if clearml_task is not None:
        callbacks.append(ClearMLSummaryCallback(task=clearml_task, config=config))

    if config.callbacks.use_rich_progress_bar and not fast_dev_run:
        callbacks.append(RichProgressBar())

    if config.callbacks.use_rich_model_summary and not fast_dev_run:
        callbacks.append(RichModelSummary(max_depth=2))

    if config.callbacks.use_swa:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.callbacks.swa_lrs,
                swa_epoch_start=config.callbacks.swa_epoch_start,
            )
        )

    return callbacks


def _build_logger(output_dir: Path) -> CSVLogger:
    return CSVLogger(
        save_dir=str(output_dir),
        name="logs",
        version="csv",
    )
