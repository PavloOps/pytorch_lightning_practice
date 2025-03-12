import torch
from torchmetrics import Metric


class FalseDiscoveryRate(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update: bool = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    allowed_tasks = ("binary", "multiclass", "multilabel")

    def __init__(
        self,
        num_classes: int,
        task: str = "multiclass",
        reduction="macro",
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if task not in self.allowed_tasks:
            raise NotImplementedError(
                f"Task '{task}' is not implemented yet. "
                f"Supported tasks: {self.allowed_tasks}"
            )

        if reduction not in ("macro", "micro", "none"):
            raise ValueError("Reduction must be one of: 'macro', 'micro', or 'none'.")

        self.task = task
        self.num_classes = num_classes
        self.reduction = reduction

        self.add_state(
            "false_positives",
            default=torch.zeros(num_classes, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "true_positives",
            default=torch.zeros(num_classes, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, prediction, target):
        # Здесь можно позже добавить обработку разных задач
        if self.task != "multiclass":
            raise NotImplementedError(
                f"Task '{self.task}' is not implemented in update(). "
                f"Only 'multiclass' currently works."
            )

        if prediction.ndim == 2:
            predicted_classes = torch.argmax(prediction, dim=1)
        else:
            predicted_classes = prediction

        predicted_classes = predicted_classes.to(target.device)

        for cls in range(self.num_classes):
            tp = torch.sum((predicted_classes == cls) & (target == cls)).float()
            fp = torch.sum((predicted_classes == cls) & (target != cls)).float()

            self.true_positives[cls] += tp
            self.false_positives[cls] += fp

    def compute(self):
        tp = self.true_positives
        fp = self.false_positives

        denominator = tp + fp
        fdr_per_class = torch.zeros_like(denominator)

        valid = denominator > 0
        fdr_per_class[valid] = fp[valid] / denominator[valid]

        if self.reduction == "macro":
            return fdr_per_class.mean()

        elif self.reduction == "micro":
            total_tp = tp.sum()
            total_fp = fp.sum()
            total_denominator = total_tp + total_fp

            if total_denominator == 0:
                return torch.tensor(0.0, device=tp.device)

            return total_fp / total_denominator

        elif self.reduction == "none":
            return fdr_per_class

    def forward(self, prediction, target):
        self.update(prediction, target)
        return self.compute()

    def plot(self, val=None, ax=None):
        return self._plot(val, ax)


if __name__ == "__main__":
    metric = FalseDiscoveryRate(num_classes=3, reduction="none")

    targets = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = torch.nn.functional.one_hot(targets, num_classes=3).float()

    metric.update(preds, targets)
    print("None:", metric.compute())  # [0., 0., 0.]

    metric = FalseDiscoveryRate(num_classes=3, reduction="none")

    targets = torch.tensor([1, 2, 1, 2])
    preds = torch.zeros((4, 3))
    preds[:, 0] = 10

    metric.update(preds, targets)
    print("None:", metric.compute())  # [1., 0., 0.]
