from __future__ import annotations

from pathlib import Path

import click
import torch
from lightning import seed_everything

from final_project.src.config import CFG
from final_project.src.data_module.two_tower_data_module import TwoTowerDataModule
from final_project.src.modeling.trainer_factory import create_trainer
from final_project.src.modeling.two_tower_model import TwoTowerModel
from final_project.src.utils.clearml import init_clearml_task, safe_close_clearml_task


def _resolve_input_path(path: Path, config: CFG) -> Path:
    if path.is_absolute():
        return path
    return Path(config.data.data_dir) / path


def _validate_input_path(path: Path, label: str, config: CFG) -> None:
    resolved_path = _resolve_input_path(path=path, config=config)
    if not resolved_path.exists():
        raise click.ClickException(
            f"{label} file not found: {resolved_path}. "
            f"Pass an existing path or update config.data.data_dir."
        )


def build_model(datamodule: TwoTowerDataModule, config: CFG) -> TwoTowerModel:
    return TwoTowerModel.from_datamodule(
        datamodule=datamodule,
        embedding_dim=config.training.embedding_dim,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        ranking_ks=config.metrics.ranking_ks,
        scheduler_name=config.optimization.scheduler_name,
        scheduler_t_max=config.optimization.scheduler_t_max,
        scheduler_eta_min=config.optimization.scheduler_eta_min,
    )


def run_training(
    config: CFG,
    output_dir: str | Path,
    train_path: str,
    val_path: str,
    test_path: str | None = None,
    fast_dev_run: bool = False,
) -> None:
    if config.trainer.accelerator == "cuda":
        torch.set_float32_matmul_precision("high")

    seed_everything(config.general.seed, workers=True)

    clearml_task = init_clearml_task(
        config=config,
        output_dir=output_dir,
        fast_dev_run=fast_dev_run,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
    )

    try:
        datamodule = TwoTowerDataModule(config=config)
        datamodule.prepare_data()
        datamodule.setup("fit")

        model = build_model(datamodule=datamodule, config=config)
        trainer = create_trainer(
            dir_path=output_dir,
            config=config,
            fast_dev_run=fast_dev_run,
            clearml_task=clearml_task,
        )

        trainer.fit(model=model, datamodule=datamodule)

        if config.data.test_file_name:
            trainer.test(model=model, datamodule=datamodule)
    finally:
        safe_close_clearml_task(clearml_task)


@click.command()
@click.option(
    "--train-path",
    type=click.Path(path_type=Path),
    default=Path("/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/raw_dataset.csv"),
    show_default=True,
)
@click.option(
    "--val-path",
    type=click.Path(path_type=Path),
    default=Path("/home/pavloops/PycharmProjects/pytorch_lightning_practice/final_project/data/raw_val_dataset.csv"),
    show_default=True,
)
@click.option("--test-path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("final_project/checkpoints/two_tower"),
)
@click.option("--fast-dev-run", is_flag=True, default=False)
def main(
    train_path: Path | None,
    val_path: Path | None,
    test_path: Path | None,
    output_dir: Path,
    fast_dev_run: bool,
) -> None:
    config = CFG()

    _validate_input_path(path=train_path, label="Train", config=config)
    _validate_input_path(path=val_path, label="Validation", config=config)

    config.data.train_file_name = str(train_path)
    config.data.val_file_name = str(val_path)
    if test_path is not None:
        _validate_input_path(path=test_path, label="Test", config=config)
        config.data.test_file_name = str(test_path)

    run_training(
        config=config,
        output_dir=output_dir,
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path) if test_path is not None else None,
        fast_dev_run=fast_dev_run,
    )


if __name__ == "__main__":
    main()
