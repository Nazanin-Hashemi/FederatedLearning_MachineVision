import pytest
import torch
import torch.nn as nn

from src.pruner import FisherPruner

STRATEGIES = ["least_sensitive", "most_sensitive", "lowest_magnitude", "highest_magnitude", "random"]


@pytest.fixture
def model_and_data():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(20, 50), nn.ReLU(), nn.Linear(50, 5))
    data = [(torch.randn(16, 20), torch.randint(0, 5, (16,))) for _ in range(4)]
    return model, data


def trained_count(mask: dict) -> int:
    return int(sum(m.sum().item() for m in mask.values()))


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("sparsity", [0.5, 0.9])
def test_trained_fraction_is_one_minus_sparsity(model_and_data, strategy, sparsity):
    model, data = model_and_data
    n_params = sum(p.numel() for p in model.parameters())
    mask = FisherPruner(sparsity, strategy).compute_mask(model, data, "cpu", num_rounds=1)

    assert abs(trained_count(mask) - (1 - sparsity) * n_params) <= 1
    assert all(set(m.unique().tolist()) <= {0.0, 1.0} for m in mask.values())


@pytest.mark.parametrize("pair", [("least_sensitive", "most_sensitive"),
                                  ("lowest_magnitude", "highest_magnitude")])
def test_opposite_rules_pick_disjoint_weights(model_and_data, pair):
    model, data = model_and_data
    low = FisherPruner(0.5, pair[0]).compute_mask(model, data, "cpu", num_rounds=1)
    high = FisherPruner(0.5, pair[1]).compute_mask(model, data, "cpu", num_rounds=1)

    assert all((low[n] * high[n]).sum() == 0 for n in low)


def test_random_mask_is_reproducible(model_and_data):
    model, data = model_and_data
    a = FisherPruner(0.5, "random", seed=1).compute_mask(model, data, "cpu")
    b = FisherPruner(0.5, "random", seed=1).compute_mask(model, data, "cpu")

    assert all(torch.equal(a[n], b[n]) for n in a)
