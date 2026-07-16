import logging
from dataclasses import asdict

import torch
from clearml import Task
from config import CFG
from lightning import Trainer
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import (
    BackboneFinetuning,
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers.logger import Logger
from modeling.debug_callbacks import ClearMLValidationDebugCallback

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def keep_learning_rate_constant(epoch: int) -> float:
    return 1.0


class ClearMLTaskLogger(Logger):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.task_logger = task.get_logger()

    @property
    def name(self):
        return self.task.name

    @property
    def version(self):
        return self.task.id

    @property
    def experiment(self):
        return self.task

    @rank_zero_only
    def log_hyperparams(self, params):
        self.task.connect(params, name="config")

    @rank_zero_only
    def log_metrics(self, metrics, step):
        iteration = 0 if step is None else step

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                if metric_value.numel() != 1:
                    continue
                metric_value = metric_value.detach().cpu().item()

            if not isinstance(metric_value, (int, float)):
                continue

            if "/" in metric_name:
                title, series = metric_name.split("/", maxsplit=1)
            else:
                title, series = "metrics", metric_name

            self.task_logger.report_scalar(
                title=title,
                series=series,
                value=metric_value,
                iteration=iteration,
            )

    @rank_zero_only
    def save(self):
        self.task.flush()

    @rank_zero_only
    def finalize(self, status):
        self.task.flush()


def create_callbacks(config: CFG, checkpoint_dir, clearml_logger, fast_dev_run=False):
    callbacks: list[Callback] = []

    if not fast_dev_run:
        callbacks.extend(
            [
                EarlyStopping(
                    monitor=config.training.monitor_metric,
                    patience=config.training.early_stopping_patience,
                    mode=config.training.monitor_mode,
                    verbose=True,
                ),
                ModelCheckpoint(
                    monitor=config.training.monitor_metric,
                    mode=config.training.monitor_mode,
                    dirpath=checkpoint_dir,
                    filename="food101-{epoch:02d}-{val_macro_f1:.4f}",
                    save_top_k=config.training.save_top_k,
                    save_last=True,
                    auto_insert_metric_name=False,
                ),
            ]
        )

    callbacks.extend(
        [
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(refresh_rate=config.trainer.progress_bar_refresh_rate),
            ClearMLValidationDebugCallback(
                config=config,
                clearml_logger=clearml_logger,
            ),
        ]
    )

    if not fast_dev_run and config.model.freeze_backbone and config.model.unfreeze_backbone_epoch is not None:
        callbacks.append(
            BackboneFinetuning(
                unfreeze_backbone_at_epoch=config.model.unfreeze_backbone_epoch,
                backbone_initial_lr=config.training.backbone_lr,
                lambda_func=keep_learning_rate_constant,
                should_align=True,
                train_bn=True,
                verbose=True,
            )
        )

    return callbacks


def create_trainer(config: CFG, checkpoint_dir, fast_dev_run=False):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer_params = asdict(config.trainer)
    trainer_params.pop("progress_bar_refresh_rate")

    if torch.cuda.is_available():
        trainer_params["precision"] = "32-true"

    clearml_task = Task.init(
        project_name=config.general.project_name,
        task_name=config.general.experiment_name,
        tags=list(config.general.tags),
    )
    clearml_logger = ClearMLTaskLogger(clearml_task)
    clearml_logger.log_hyperparams(asdict(config))

    return Trainer(
        **trainer_params,
        fast_dev_run=fast_dev_run,
        logger=clearml_logger,
        callbacks=create_callbacks(config, checkpoint_dir, clearml_logger, fast_dev_run),
    )
