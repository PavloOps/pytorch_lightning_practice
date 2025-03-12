from unittest import TestCase, main

import torch

from src.custom_metric import FalseDiscoveryRate


class TestFalseDiscoveryRate(TestCase):
    def setUp(self):
        self.num_classes = 3

    def test_perfect_predictions_none(self):
        metric = FalseDiscoveryRate(num_classes=self.num_classes, reduction="none")

        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        predictions = torch.nn.functional.one_hot(
            targets, num_classes=self.num_classes
        ).float()

        metric.update(predictions, targets)
        result = metric.compute()

        expected = torch.zeros(self.num_classes, device=result.device)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    def test_all_wrong_predictions_none(self):
        metric = FalseDiscoveryRate(num_classes=self.num_classes, reduction="none")

        targets = torch.tensor([1, 2, 1, 2])
        predictions = torch.zeros((4, self.num_classes))
        predictions[:, 0] = 10

        metric.update(predictions, targets)
        result = metric.compute()

        # Class 0: 4 FP, 0 TP => FDR = 1
        # Classes 1 и 2 were not chosen, denominator = 0 => FDR = 0
        expected = torch.tensor([1.0, 0.0, 0.0], device=result.device)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    def test_partial_predictions_none(self):
        metric = FalseDiscoveryRate(num_classes=self.num_classes, reduction="none")

        targets = torch.tensor([0, 1, 1, 2, 2, 2])
        preds_classes = torch.tensor([0, 2, 1, 2, 0, 2])

        predictions = torch.nn.functional.one_hot(
            preds_classes, num_classes=self.num_classes
        ).float()

        metric.update(predictions, targets)
        result = metric.compute()

        # Class 0:
        #   TP: pred=0 & target=0 => 1
        #   FP: pred=0 & target!=0 => 1 (index 4)
        #   FDR = FP / (TP + FP) = 1 / (1 + 1) = 0.5
        # Class 1:
        #   TP: pred=1 & target=1 => 1 (index 2)
        #   FP: pred=1 & target!=1 => 0
        #   FDR = 0 / (1 + 0) = 0
        # Class 2:
        #   TP: pred=2 & target=2 => 2 (index 3, 5)
        #   FP: pred=2 & target!=2 => 1 (index 1)
        #   FDR = 1 / (2 + 1) = 1/3
        expected = torch.tensor([0.5, 0.0, 1 / 3], device=result.device)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    def test_no_predictions_for_class_none(self):
        metric = FalseDiscoveryRate(num_classes=self.num_classes, reduction="none")

        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        predictions = torch.zeros((6, self.num_classes))
        predictions[:, 0] = 10

        metric.update(predictions, targets)
        result = metric.compute()

        # Class 0: TP = 2, FP = 4 => FDR = 4 / (2 + 4) = 2/3
        # Class 1 и 2 aren't predicted => denominator = 0 => FDR = 0
        expected = torch.tensor([2 / 3, 0.0, 0.0], device=result.device)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    def test_micro_average(self):
        metric = FalseDiscoveryRate(num_classes=self.num_classes, reduction="micro")

        targets = torch.tensor([0, 1, 1, 2, 2, 2])
        preds_classes = torch.tensor([0, 2, 1, 2, 0, 2])

        predictions = torch.nn.functional.one_hot(
            preds_classes, num_classes=self.num_classes
        ).float()

        metric.update(predictions, targets)
        result = metric.compute()

        # Sum of TP и FP:
        # Class 0: TP=1, FP=1
        # Class 1: TP=1, FP=0
        # Class 2: TP=2, FP=1
        total_tp = 1 + 1 + 2  # = 4
        total_fp = 1 + 0 + 1  # = 2
        expected_value = total_fp / (total_tp + total_fp)  # 2 / (4 + 2) = 1/3
        expected = torch.tensor(
            expected_value, device=result.device, dtype=result.dtype
        )
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    def test_macro_average(self):
        metric = FalseDiscoveryRate(num_classes=self.num_classes, reduction="macro")
        targets = torch.tensor([0, 1, 1, 2, 2, 2])
        predicteds_classes = torch.tensor([0, 2, 1, 2, 0, 2])
        predictions = torch.nn.functional.one_hot(
            predicteds_classes, num_classes=self.num_classes
        ).float()

        metric.update(predictions, targets)
        result = metric.compute()
        # [0.5, 0.0, 1/3] => macro mean = (0.5 + 0.0 + 1/3)/3
        expected = (0.5 + 0.0 + 1 / 3) / 3
        expected_tensor = torch.tensor(expected, device=result.device)

        torch.testing.assert_close(result, expected_tensor, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    main()
