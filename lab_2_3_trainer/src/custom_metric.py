import torch
from torchmetrics import Metric


class FalseDiscoveryRate(Metric):
    """Compute False Discovery Rate (FDR = FP / (FP + TP)) for multiclass tasks."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    allowed_tasks = ("binary", "multiclass", "multilabel")

    def __init__(
        self, num_classes: int, task: str = "multiclass", reduction: str = "macro"
    ):
        super().__init__()

        if task not in self.allowed_tasks:
            raise NotImplementedError(
                f"Unsupported task '{task}'. Supported: {self.allowed_tasks}"
            )
        if reduction not in ("macro", "micro", "none"):
            raise ValueError("Reduction must be one of: 'macro', 'micro', or 'none'.")

        self.task = task
        self.num_classes = num_classes
        self.reduction = reduction

        self.add_state(
            "false_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "true_positives", default=torch.zeros(num_classes), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim == 2:
            preds = torch.argmax(preds, dim=1)
        preds = preds.to(target.device)

        for cls in range(self.num_classes):
            mask_pred = preds == cls
            tp = (mask_pred & (target == cls)).sum().float()
            fp = (mask_pred & (target != cls)).sum().float()
            self.true_positives[cls] += tp
            self.false_positives[cls] += fp

    def compute(self):
        tp, fp = self.true_positives, self.false_positives
        denom = tp + fp
        fdr = torch.zeros_like(denom)
        valid = denom > 0
        fdr[valid] = fp[valid] / denom[valid]

        if self.reduction == "macro":
            return fdr.mean()
        elif self.reduction == "micro":
            total_tp, total_fp = tp.sum(), fp.sum()
            total_denom = total_tp + total_fp
            return total_fp / total_denom if total_denom > 0 else torch.tensor(0.0)
        return fdr

    def plot(self, val=None, ax=None):
        return self._plot(val, ax)


if __name__ == "__main__":
    metric = FalseDiscoveryRate(num_classes=3, reduction="none")

    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    predictions = torch.nn.functional.one_hot(targets, num_classes=3).float()

    metric.update(predictions, targets)
    print("None:", metric.compute())  # [0., 0., 0.]

    metric = FalseDiscoveryRate(num_classes=3, reduction="none")

    targets = torch.tensor([1, 2, 1, 2])
    predictions = torch.zeros((4, 3))
    predictions[:, 0] = 10

    metric.update(predictions, targets)
    print("None:", metric.compute())  # [1., 0., 0.]
