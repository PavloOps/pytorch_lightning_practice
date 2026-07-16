import csv
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from config import CFG
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Food101ErrorAnalyzer:
    def __init__(self, config: CFG, datamodule):
        self.config = config
        self.datamodule = datamodule
        self.hard_cases_dir = Path(config.data.hard_cases_dir)
        self.manifest_path = Path(config.data.hard_cases_manifest_path)
        self.confusion_matrix_path = Path(config.data.confusion_matrix_path)

    def run(self, model):
        if self.datamodule.val_dataset is None:
            self.datamodule.setup(stage="fit")

        records = self.collect_validation_predictions(model)
        self.save_confusion_matrix(records)
        self.save_hard_cases(records)

    def collect_validation_predictions(self, model):
        device = next(model.parameters()).device
        model.eval()
        records = []

        with torch.no_grad():
            for batch in tqdm(self.datamodule.val_dataloader(), desc="Collecting validation predictions"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)

                for row_idx in range(labels.shape[0]):
                    true_idx = labels[row_idx].item()
                    pred_idx = predictions[row_idx].item()
                    records.append(
                        {
                            "image_path": batch["image_path"][row_idx],
                            "true_idx": true_idx,
                            "pred_idx": pred_idx,
                            "true_class": self.datamodule.classes[true_idx],
                            "pred_class": self.datamodule.classes[pred_idx],
                            "confidence": confidences[row_idx].item(),
                        }
                    )

        return records

    def save_confusion_matrix(self, records):
        true_labels = [record["true_idx"] for record in records]
        pred_labels = [record["pred_idx"] for record in records]
        class_names = self.datamodule.classes
        matrix = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))

        figure, axis = plt.subplots(figsize=(24, 24))
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
        display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation="vertical", values_format="d")
        axis.set_title("Food-101 Validation Confusion Matrix")
        figure.tight_layout()

        self.confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.confusion_matrix_path, dpi=180)
        plt.close(figure)
        logger.info("Confusion matrix is saved to %s.", self.confusion_matrix_path)

    def save_hard_cases(self, records):
        wrong_records = [record for record in records if record["true_idx"] != record["pred_idx"]]
        grouped_records = defaultdict(list)

        for record in wrong_records:
            grouped_records[(record["true_class"], record["pred_class"])].append(record)

        sorted_groups = sorted(
            grouped_records.items(),
            key=lambda item: (len(item[1]), max(record["confidence"] for record in item[1])),
            reverse=True,
        )

        self.hard_cases_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        group_limit = self.config.training.hard_confusion_pairs
        sample_limit = self.config.training.hard_case_samples_per_pair

        for (true_class, pred_class), group_records in sorted_groups[:group_limit]:
            group_name = f"{self.safe_name(true_class)}__pred_{self.safe_name(pred_class)}"
            group_dir = self.hard_cases_dir / group_name
            group_dir.mkdir(parents=True, exist_ok=True)
            group_records = sorted(group_records, key=lambda record: record["confidence"], reverse=True)

            for sample_idx, record in enumerate(group_records[:sample_limit]):
                source_path = Path(record["image_path"])
                sample_path = group_dir / f"{sample_idx:02d}_{source_path.name}"
                shutil.copy2(source_path, sample_path)
                manifest_rows.append(
                    {
                        "group_name": group_name,
                        "true_class": true_class,
                        "baseline_pred_class": pred_class,
                        "baseline_confidence": f"{record['confidence']:.6f}",
                        "source_image_path": str(source_path),
                        "sample_path": str(sample_path),
                    }
                )

        with open(self.manifest_path, "w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=[
                    "group_name",
                    "true_class",
                    "baseline_pred_class",
                    "baseline_confidence",
                    "source_image_path",
                    "sample_path",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        logger.info(
            "Saved %s hard validation samples across %s confusion groups to %s.",
            len(manifest_rows),
            min(len(sorted_groups), group_limit),
            self.hard_cases_dir,
        )

    @staticmethod
    def safe_name(class_name):
        return class_name.replace("/", "_").replace(" ", "_")
