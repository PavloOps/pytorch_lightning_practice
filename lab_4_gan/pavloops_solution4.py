import contextlib
import io
from dataclasses import asdict

from lab_4_gan.config import CFG
from lab_4_gan.src.mnist_trainer import create_trainer
from lab_4_gan.src.gan_network import GAN
from lab_4_gan.src.mnist_data_module import MNISTDataset, MNISTLightning
from lightning import seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def run_experiment(config, epoch=10, debug_samples_epoch=1, need_dev_run=True):
    seed_everything(config.general.seed)
    config.trainer.max_epochs = epoch

    dataset = MNISTLightning(config=config)
    model = GAN(config=config)

    if need_dev_run:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
            ):
                trainer = create_trainer(dir_path="./", config=config, fast_dev_run=True)
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
    run_experiment(config=curr_config)
