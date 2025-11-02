import contextlib
import io
from dataclasses import asdict

from clearml import Task
from lightning import seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from src.config import CFG
from src.gan_network import GAN
from src.mnist_data_module import MNISTLightning
from src.mnist_trainer import create_trainer


def run_experiment(
    config, clearml_logger, epoch=10, debug_samples_epoch=1, need_dev_run=True
):
    seed_everything(config.general.seed)
    config.trainer.max_epochs = epoch
    config.trainer.debug_samples_epoch = debug_samples_epoch

    dataset = MNISTLightning(config=config)
    model = GAN(config=config, clearml_logger=clearml_logger)

    if need_dev_run:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                trainer = create_trainer(
                    dir_path="./", config=config, fast_dev_run=True
                )
                trainer.fit(model, dataset)
            print("Debug run has been finished.")
        except (MisconfigurationException, RuntimeError, ValueError) as e:
            print(e)
        except Exception as e:
            print("Unexpected error:", e)

    trainer = create_trainer(dir_path="./", config=config)
    trainer.fit(model, dataset)
    dataset.teardown()


if __name__ == "__main__":
    curr_config = CFG()
    curr_config_dict = asdict(curr_config)

    task = Task.init(
        project_name="Машинное обучение с помощью ClearML и Pytorch Lighting",
        task_name="Lab 4 GAN & Debug Pictures",
        tags=["Lab4", "MNIST", "digits", "GAN"],
    )

    task.execute_remotely(queue_name="default")

    curr_clearml_logger = task.get_logger()

    for sub_config_name in curr_config_dict:
        task.connect(
            mutable=curr_config_dict[sub_config_name],
            name=sub_config_name,
        )

    run_experiment(config=curr_config, clearml_logger=curr_clearml_logger)

    task.close()
