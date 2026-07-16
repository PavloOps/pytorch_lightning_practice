import csv
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ClearMLValidationDebugCallback(Callback):
    def __init__(self, config, clearml_logger):
        super().__init__()
        self.config = config
        self.clearml_logger = clearml_logger
        self.task_logger = clearml_logger.task_logger

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self.config.training.debug_samples_epoch <= 0:
            return
        if self._should_skip_epoch(pl_module.current_epoch):
            return
        if trainer.datamodule is None or trainer.datamodule.val_dataset is None:
            return

        if self.hard_cases_manifest_exists():
            self.log_hard_case_groups(trainer, pl_module)
            return

        batch = self.get_validation_batch(trainer)
        self.log_debug_batch(batch, trainer, pl_module)

    def _should_skip_epoch(self, epoch):
        return epoch % self.config.training.debug_samples_epoch != 0

    @staticmethod
    def get_validation_batch(trainer):
        dataloaders = trainer.val_dataloaders
        dataloader = dataloaders[0] if isinstance(dataloaders, list) else dataloaders
        return next(iter(dataloader))

    def hard_cases_manifest_exists(self):
        return Path(self.config.data.hard_cases_manifest_path).is_file()

    def log_hard_case_groups(self, trainer, pl_module):
        rows_by_group = self.read_hard_cases_manifest()
        class_names = self.get_class_names(trainer)
        class_to_idx = getattr(trainer.datamodule, "class_to_idx", None)

        if not class_names or class_to_idx is None:
            logger.warning("Hard cases debug skipped: class names are not available.")
            return

        logged_groups = 0
        for group_name, rows in rows_by_group.items():
            rows = rows[: self.config.training.hard_case_samples_per_pair]
            images = []
            labels = []

            for row in rows:
                image_path = Path(row["sample_path"])
                if not image_path.is_file():
                    logger.warning("Hard case image is not found: %s.", image_path)
                    continue

                image = Image.open(image_path).convert("RGB")
                images.append(trainer.datamodule.eval_transform(image))
                labels.append(class_to_idx[row["true_class"]])

            if not images:
                continue

            images = torch.stack(images).to(pl_module.device)
            labels = torch.tensor(labels, device=pl_module.device)
            probabilities, predictions, gradcams = self.get_predictions_and_gradcams(pl_module, images)
            debug_images = []

            for image_idx in range(images.shape[0]):
                debug_images.append(
                    self.build_debug_image(
                        image=images[image_idx],
                        gradcam=gradcams[image_idx],
                        probabilities=probabilities[image_idx],
                        prediction=predictions[image_idx],
                        label=labels[image_idx].cpu(),
                        class_names=class_names,
                    )
                )

            self.task_logger.report_image(
                title="Debug/Hard Confusions Grad-CAM",
                series=group_name,
                iteration=pl_module.current_epoch,
                image=self.build_debug_grid(debug_images),
            )
            logged_groups += 1

        logger.info("Logged %s hard validation debug groups to ClearML.", logged_groups)

    def read_hard_cases_manifest(self):
        rows_by_group = defaultdict(list)

        with open(self.config.data.hard_cases_manifest_path, newline="", encoding="utf-8") as manifest_file:
            for row in csv.DictReader(manifest_file):
                rows_by_group[row["group_name"]].append(row)

        return dict(list(rows_by_group.items())[: self.config.training.hard_confusion_pairs])

    def log_debug_batch(self, batch, trainer, pl_module):
        images = batch["image"][: self.config.training.num_debug_samples].to(pl_module.device)
        labels = batch["label"][: self.config.training.num_debug_samples].to(pl_module.device)
        class_names = self.get_class_names(trainer)

        probabilities, predictions, gradcams = self.get_predictions_and_gradcams(pl_module, images)

        for image_idx in range(images.shape[0]):
            debug_image = self.build_debug_image(
                image=images[image_idx],
                gradcam=gradcams[image_idx],
                probabilities=probabilities[image_idx],
                prediction=predictions[image_idx],
                label=labels[image_idx],
                class_names=class_names,
            )
            self.task_logger.report_image(
                title="Debug/Validation Grad-CAM",
                series=f"sample_{image_idx}",
                iteration=pl_module.current_epoch,
                image=debug_image,
            )

        logger.info(
            "Logged %s validation debug samples to ClearML.",
            images.shape[0],
        )

    def get_predictions_and_gradcams(self, pl_module, images):
        images = images.detach().clone().requires_grad_(True)
        activations = []
        gradients = []

        def forward_hook(module, inputs, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        target_layer = pl_module.backbone[-1]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        was_training = pl_module.training
        pl_module.eval()

        try:
            with torch.enable_grad():
                pl_module.zero_grad(set_to_none=True)
                logits = pl_module(images)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                target_scores = logits[torch.arange(logits.shape[0], device=logits.device), predictions]
                target_scores.sum().backward()

            gradcams = self.calculate_gradcams(
                activations=activations[-1],
                gradients=gradients[-1],
                image_size=images.shape[-2:],
            )
        finally:
            forward_handle.remove()
            backward_handle.remove()
            pl_module.zero_grad(set_to_none=True)
            if was_training:
                pl_module.train()

        return (
            probabilities.detach().cpu(),
            predictions.detach().cpu(),
            gradcams.detach().cpu(),
        )

    @staticmethod
    def calculate_gradcams(activations, gradients, image_size):
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        gradcams = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        gradcams = F.interpolate(
            gradcams,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )

        batch_size = gradcams.shape[0]
        gradcams = gradcams.view(batch_size, -1)
        gradcam_min = gradcams.min(dim=1, keepdim=True).values
        gradcam_max = gradcams.max(dim=1, keepdim=True).values
        gradcams = (gradcams - gradcam_min) / (gradcam_max - gradcam_min + 1e-8)
        return gradcams.view(batch_size, 1, *image_size)

    def build_debug_image(
        self,
        image,
        gradcam,
        probabilities,
        prediction,
        label,
        class_names,
    ):
        original = self.tensor_to_image(image)
        heatmap = self.gradcam_to_image(gradcam)
        overlay = Image.blend(
            original,
            heatmap,
            alpha=self.config.training.gradcam_alpha,
        )
        text_panel = self.build_text_panel(
            probabilities=probabilities,
            prediction=prediction,
            label=label,
            class_names=class_names,
            height=original.height,
        )

        canvas = Image.new(
            "RGB",
            (original.width * 2 + text_panel.width, original.height),
            color=(255, 255, 255),
        )
        canvas.paste(original, (0, 0))
        canvas.paste(overlay, (original.width, 0))
        canvas.paste(text_panel, (original.width * 2, 0))
        return canvas

    def tensor_to_image(self, image):
        mean = torch.tensor(self.config.transform.normalize_mean).view(3, 1, 1)
        std = torch.tensor(self.config.transform.normalize_std).view(3, 1, 1)
        image = image.detach().cpu() * std + mean
        image = image.clamp(0, 1).permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    @staticmethod
    def gradcam_to_image(gradcam):
        gradcam = gradcam.squeeze(0).numpy()
        heatmap = plt.get_cmap("jet")(gradcam)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        return Image.fromarray(heatmap)

    def build_text_panel(self, probabilities, prediction, label, class_names, height):
        panel_width = 360
        background_color = (229, 245, 232) if prediction.item() == label.item() else (255, 237, 213)
        panel = Image.new("RGB", (panel_width, height), color=background_color)
        draw = ImageDraw.Draw(panel)

        true_class = self.get_class_name(class_names, label.item())
        pred_class = self.get_class_name(class_names, prediction.item())
        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(self.config.training.debug_top_k, probabilities.shape[0]),
        )

        y = 12
        lines = [
            f"true: {true_class}",
            f"pred: {pred_class}",
            f"confidence: {probabilities[prediction].item():.3f}",
            "",
            "top predictions:",
        ]

        for line in lines:
            draw.text((12, y), line, fill=(0, 0, 0))
            y += 22

        for prob, class_idx in zip(top_probs, top_indices):
            class_name = self.get_class_name(class_names, class_idx.item())
            draw.text(
                (12, y),
                f"{class_name}: {prob.item():.3f}",
                fill=(0, 0, 0),
            )
            y += 22

        return panel

    @staticmethod
    def build_debug_grid(images):
        if len(images) == 1:
            return images[0]

        columns = 2
        rows = int(np.ceil(len(images) / columns))
        cell_width = max(image.width for image in images)
        cell_height = max(image.height for image in images)
        grid = Image.new("RGB", (columns * cell_width, rows * cell_height), color=(255, 255, 255))

        for image_idx, image in enumerate(images):
            row_idx = image_idx // columns
            column_idx = image_idx % columns
            grid.paste(image, (column_idx * cell_width, row_idx * cell_height))

        return grid

    @staticmethod
    def get_class_names(trainer):
        classes = getattr(trainer.datamodule, "classes", None)
        if classes is None:
            return []
        return classes

    @staticmethod
    def get_class_name(class_names, class_idx):
        if class_names and class_idx < len(class_names):
            return class_names[class_idx]
        return str(class_idx)
