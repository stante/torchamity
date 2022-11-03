import pytest
import torch
from torchamity.metrics import Accuracy


@pytest.fixture
def tensors75():
    return [torch.Tensor([1, 1, 1, 1]), torch.Tensor([0, 1, 1, 1])]


@pytest.fixture
def tensors25():
    return [torch.Tensor([0, 0, 0, 0]), torch.Tensor([1, 1, 1, 0])]


@pytest.fixture
def tensors0():
    return [torch.Tensor([0, 0, 0, 0]), torch.Tensor([1, 1, 1, 1])]


def test_accuracy_result_50(tensors75, tensors25):
    acc = Accuracy()
    acc.update(tensors25[0], tensors25[1])
    acc.update(tensors75[0], tensors75[1])
    assert acc.result() == 0.5


def test_accuracy_result_0(tensors0):
    acc = Accuracy()
    acc.update(tensors0[0], tensors0[1])
    assert acc.result() == 0


def test_accuracy_result_without_update():
    acc = Accuracy()
    assert acc.result() == 0
