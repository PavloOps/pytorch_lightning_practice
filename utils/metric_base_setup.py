import torch
from torchmetrics import Metric


class ExampleMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            name="counter",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, predictions, targets):
        self.counter = self.counter + torch.sum(predictions == targets)

    def compute(self):
        return self.counter


m = ExampleMetric()

print(m.counter)

m.update(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4])) # can be replaced with call method

print(m.counter)

print(m.compute())

m.reset()

print(m.counter)

m(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 4])) # Object as function

print(m.counter)
