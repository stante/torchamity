import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchamity.learner import Learner
from torchamity.metrics import Accuracy


@pytest.fixture
def model():
    model = nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    return model


@pytest.fixture
def data():
    x_test = torch.randn((32, 16))
    y_test = torch.randint(0, 4, (32, 1)).float()

    x_val = torch.randn((16, 16))
    y_val = torch.randint(0, 4, (16, 1)).float()

    return [TensorDataset(x_test, y_test), TensorDataset(x_val, y_val)]


@pytest.fixture
def optim(model):
    return torch.optim.Adam(model.parameters(), lr=0.0001)


@pytest.fixture
def loss():
    return nn.CrossEntropyLoss()


@pytest.fixture
def learner(data, optim, loss, model):
    return Learner(data[0], data[1], optim, loss, model)


def test_learner_epochs(learner):
    result = learner.fit(3)

    assert len(result['loss']) == 3


def test_learner_val_metrics(learner):
    result = learner.fit(3, val_metrics=[Accuracy()])

    assert len(result['val_acc']) == 3
