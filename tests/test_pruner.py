import pytest
import torch
import torch.nn as nn

from src.pruner import FisherPruner

STRATEGIES = ["least_sensitive", "most_sensitive", "lowest_magnitude", "highest_magnitude", "random"]
DENSE = {"cls_token", "pos_embed"}


class TinyViTLike(nn.Module):
    """Tiny stand-in with the parameter names that matter: cls_token, pos_embed, LayerNorm, Linear."""
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(0.02 * torch.randn(20))
        self.pos_embed = nn.Parameter(0.02 * torch.randn(20))
        self.norm = nn.LayerNorm(20)
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.norm(x + self.cls_token + self.pos_embed)
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def model_and_data():
    torch.manual_seed(0)
    model = TinyViTLike()
    data = [(torch.randn(16, 20), torch.randint(0, 5, (16,))) for _ in range(4)]
    return model, data


def n_scored(model) -> int:
    return sum(p.numel() for name, p in model.named_parameters() if name not in DENSE)


def trained_count(mask: dict) -> int:
    return int(sum(m.sum().item() for m in mask.values()))


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("sparsity", [0.5, 0.9])
@pytest.mark.parametrize("rounds", [1, 4])
def test_trained_fraction_is_one_minus_sparsity(model_and_data, strategy, sparsity, rounds):
    model, data = model_and_data
    pruner = FisherPruner(sparsity, strategy)
    mask = pruner.compute_mask(model, data, "cpu", num_rounds=rounds)

    assert abs(trained_count(mask) - (1 - sparsity) * n_scored(model)) <= 1
    assert pruner.trained_fraction == pytest.approx(1 - sparsity, abs=1 / n_scored(model))
    assert all(set(m.unique().tolist()) <= {0.0, 1.0} for m in mask.values())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_dense_params_are_not_masked(model_and_data, strategy):
    model, data = model_and_data
    mask = FisherPruner(0.5, strategy).compute_mask(model, data, "cpu", num_rounds=2)

    assert DENSE.isdisjoint(mask)
    assert set(mask) == {name for name, _ in model.named_parameters()} - DENSE


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


@pytest.mark.parametrize("strategy", ["least_sensitive", "most_sensitive"])
def test_progressive_schedule(model_and_data, strategy):
    model, data = model_and_data
    pruner = FisherPruner(0.9, strategy)
    pruner.compute_mask(model, data, "cpu", num_rounds=4)

    expected = [0.1 ** ((r + 1) / 4) for r in range(4)]
    assert pruner.round_fractions == pytest.approx(expected, abs=1 / n_scored(model))


@pytest.mark.parametrize("train_lowest", [True, False])
def test_eligible_restricts_choice(train_lowest):
    torch.manual_seed(0)
    pruner = FisherPruner()
    scores = {"a": torch.rand(100), "b": torch.rand(50)}
    eligible = {"a": (torch.rand(100) > 0.5).float(), "b": (torch.rand(50) > 0.5).float()}

    mask = pruner._build_mask(scores, train_lowest, keep_fraction=0.2, eligible=eligible)

    assert trained_count(mask) == 30
    assert all(((mask[n] == 1) <= (eligible[n] == 1)).all() for n in mask)


def test_multi_round_changes_mask(model_and_data):
    model, data = model_and_data
    one = FisherPruner(0.9, "least_sensitive").compute_mask(model, data, "cpu", num_rounds=1)
    four = FisherPruner(0.9, "least_sensitive").compute_mask(model, data, "cpu", num_rounds=4)

    assert any(not torch.equal(one[n], four[n]) for n in one)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_weights_restored_after_calibration(model_and_data, strategy):
    model, data = model_and_data
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    FisherPruner(0.9, strategy).compute_mask(model, data, "cpu", num_rounds=4)

    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    assert all(p.grad is None or (p.grad == 0).all() for p in model.parameters())
