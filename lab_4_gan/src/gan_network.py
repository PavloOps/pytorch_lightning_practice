from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.init as init
from clearml import Task
from lightning import LightningModule
from torchvision import transforms
from torchvision.utils import make_grid

from .config import CFG


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise_vector):
        return self.model(noise_vector)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_image):
        return self.model(input_image)


class GAN(LightningModule):
    def __init__(self, config: CFG, clearml_logger: Task.logger):
        super().__init__()
        self.config = config
        self.clearml_logger = clearml_logger
        self.save_hyperparameters(asdict(config))

        self.batch_size = config.training.batch_size
        self.noise_dim = config.training.noise_dim
        self.lr = self.config.training.lr
        self.betas = self.config.training.betas

        self.generator = Generator(self.noise_dim)
        self.discriminator = Discriminator()
        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)
        self.criterion = nn.BCELoss()

        self.automatic_optimization = False

        self.fixed_noise = torch.randn(5, self.noise_dim)
        self.resize_transform = transforms.Resize((128, 128))

    def forward(self, noise_vector):
        return self.generator(noise_vector)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr * 0.25, betas=(0.6, 0.999)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Discriminator
        noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
        fake_images = self(noise).detach()
        true_images = batch[0]

        real_labels = torch.ones((self.batch_size, 1), device=self.device)
        fake_labels = torch.zeros((self.batch_size, 1), device=self.device)

        real_output = self.discriminator(true_images)
        fake_output = self.discriminator(fake_images)

        loss_d_real = self.criterion(real_output, real_labels)
        loss_d_fake = self.criterion(fake_output, fake_labels)
        loss_d = (loss_d_real + loss_d_fake) / 2

        opt_d.zero_grad()
        self.manual_backward(loss_d)
        opt_d.step()

        # Generator
        noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
        fake_images = self(noise)
        fake_output = self.discriminator(fake_images)
        loss_g = self.criterion(fake_output, real_labels)

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        self.log("train/loss_discriminator", loss_d, prog_bar=True)
        self.log("train/loss_generator", loss_g, prog_bar=True)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Use fixed noise
        fake_images = self(self.fixed_noise.to(self.device))
        fake_images = (fake_images + 1) / 2

        true_images = batch[0]
        batch_size = true_images.size(0)

        real_labels = torch.ones((batch_size, 1), device=self.device)
        fake_labels = torch.zeros((batch_size, 1), device=self.device)

        real_output = self.discriminator(true_images)
        fake_output = self.discriminator(fake_images[:batch_size])

        loss_d_real = self.criterion(real_output, real_labels)
        loss_d_fake = self.criterion(fake_output, fake_labels)
        loss_d = (loss_d_real + loss_d_fake) / 2

        fake_output_for_g = self.discriminator(fake_images[:batch_size])
        loss_g = self.criterion(fake_output_for_g, real_labels)

        self.log("val/loss_discriminator", loss_d, prog_bar=True, on_epoch=True)
        self.log("val/loss_generator", loss_g, prog_bar=True, on_epoch=True)

        return {"val_loss": (loss_d + loss_g) / 2}

    @torch.no_grad()
    def on_validation_epoch_end(self):
        if self.current_epoch % self.config.training.debug_samples_epoch == 0:
            fake_images = self.generator(self.fixed_noise.to(self.device))
            fake_images = (fake_images + 1) / 2  # [-1,1] → [0,1]

            # 128×128
            fake_images_resized = self.resize_transform(fake_images)
            grid = make_grid(fake_images_resized, nrow=5, padding=4, normalize=False)
            np_image = grid.cpu().numpy().transpose(1, 2, 0)

            self.clearml_logger.report_image(
                title="Generated Samples",
                series="debug_samples",
                iteration=self.current_epoch,
                image=np_image,
            )
