from __future__ import annotations

from pathlib import Path

import click
from lightning import seed_everything

from final_project.src.config import CFG
from final_project.src.data_module.two_tower_data_module import TwoTowerDataModule
from final_project.src.modeling.trainer_factory import create_trainer
from final_project.src.modeling.two_tower_model import TwoTowerModel


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
    fast_dev_run: bool = False,
) -> None:
    seed_everything(config.general.seed, workers=True)

    datamodule = TwoTowerDataModule(config=config)
    datamodule.prepare_data()
    datamodule.setup("fit")

    model = build_model(datamodule=datamodule, config=config)
    trainer = create_trainer(
        dir_path=output_dir,
        config=config,
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model=model, datamodule=datamodule)

    if config.data.test_file_name:
        trainer.test(model=model, datamodule=datamodule)


@click.command()
@click.option("--train-path", type=click.Path(path_type=Path), default=None)
@click.option("--val-path", type=click.Path(path_type=Path), default=None)
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

    if train_path is not None:
        config.data.train_file_name = str(train_path)
    if val_path is not None:
        config.data.val_file_name = str(val_path)
    if test_path is not None:
        config.data.test_file_name = str(test_path)

    run_training(
        config=config,
        output_dir=output_dir,
        fast_dev_run=fast_dev_run,
    )


if __name__ == "__main__":
    main()
