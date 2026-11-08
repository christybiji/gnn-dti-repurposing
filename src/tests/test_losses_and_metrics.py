"""
Standalone, CI-runnable tests.

The notebook's Section 10 has several `assert` statements checking the
censored loss and the concordance index, but they run inline, once, as part
of a 3,700-line linear execution -- there's no way to run "just the loss
tests" without downloading BindingDB and executing everything before them.

This file pulls those checks out into real pytest tests (so they run in
isolation, in CI, in milliseconds) and adds tests for the new Tobit loss
and for edge cases the notebook doesn't currently check.

Run with:
    pytest tests/test_losses_and_metrics.py -v
"""
import math

import numpy as np
import pytest
import torch

from losses import censored_hinge_loss, tobit_censored_loss, LearnableSigma
from metrics import concordance_index

FLOOR = 5.0  # matches the notebook's PKD_FLOOR for Kd_ceiling = 10,000 nM


# --------------------------------------------------------------------------- #
# Tests mirroring the notebook's own inline assertions (Section 10)
# --------------------------------------------------------------------------- #

def test_hinge_loss_perfect_prediction_is_zero():
    y = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0] * 20)
    censored = torch.zeros(len(y), dtype=torch.bool)
    loss = censored_hinge_loss(y, y, censored, floor=FLOOR)
    assert loss.item() < 1e-9


def test_hinge_loss_below_floor_costs_nothing():
    # Predicting weaker than the detection limit for a censored pair is not
    # wrong -- the true value is "at least this weak" -- so it must be free.
    pred = torch.tensor([3.0])
    y = torch.tensor([5.0])
    censored = torch.tensor([True])
    loss = censored_hinge_loss(pred, y, censored, floor=FLOOR)
    assert loss.item() == 0.0


def test_hinge_loss_claiming_binding_at_nonbinder_costs_something():
    pred = torch.tensor([7.0])
    y = torch.tensor([5.0])
    censored = torch.tensor([True])
    loss = censored_hinge_loss(pred, y, censored, floor=FLOOR)
    assert loss.item() > 0.0


def test_concordance_index_perfect_ranking():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 50)
    ci = concordance_index(y, y)
    assert ci > 0.999


def test_concordance_index_inverted_ranking():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 50)
    ci = concordance_index(y, -y)
    assert ci < 0.001


# --------------------------------------------------------------------------- #
# New tests: edge cases the notebook does not currently check
# --------------------------------------------------------------------------- #

def test_hinge_loss_all_censored_batch():
    pred = torch.tensor([3.0, 4.0, 6.0])
    y = torch.full((3,), FLOOR)
    censored = torch.tensor([True, True, True])
    loss = censored_hinge_loss(pred, y, censored, floor=FLOOR)
    # Only the third prediction (6.0 > floor) should contribute.
    expected = ((6.0 - FLOOR) ** 2) / 3
    assert math.isclose(loss.item(), expected, rel_tol=1e-5)


def test_hinge_loss_all_observed_batch_matches_mse():
    pred = torch.tensor([5.5, 6.5, 7.5])
    y = torch.tensor([5.0, 7.0, 7.0])
    censored = torch.zeros(3, dtype=torch.bool)
    loss = censored_hinge_loss(pred, y, censored, floor=FLOOR)
    expected = ((pred - y) ** 2).mean()
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-6)


def test_concordance_index_too_few_points_returns_nan():
    assert math.isnan(concordance_index(np.array([1.0]), np.array([1.0])))


def test_concordance_index_all_tied_labels_returns_nan():
    y = np.array([5.0, 5.0, 5.0])
    p = np.array([1.0, 2.0, 3.0])
    assert math.isnan(concordance_index(y, p))


# --------------------------------------------------------------------------- #
# Tobit loss: the upgrade the notebook names but does not implement
# --------------------------------------------------------------------------- #

def test_tobit_loss_perfect_prediction_is_near_minimal():
    """A tight, correct fit should have much lower loss than a poor fit,
    at the loss's own optimal sigma."""
    y = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0] * 20)
    censored = torch.zeros(len(y), dtype=torch.bool)

    good_log_sigma = torch.tensor(math.log(0.05))
    bad_log_sigma = torch.tensor(math.log(0.05))

    good_pred = y.clone()
    bad_pred = y + 3.0

    good_loss = tobit_censored_loss(good_pred, y, censored, good_log_sigma, FLOOR)
    bad_loss = tobit_censored_loss(bad_pred, y, censored, bad_log_sigma, FLOOR)
    assert good_loss.item() < bad_loss.item()


def test_tobit_loss_below_floor_costs_little_as_sigma_shrinks():
    """As sigma -> small, a confident prediction well below the floor for a
    censored row should cost very little (survival probability -> 1)."""
    pred = torch.tensor([2.0])  # well below floor=5.0
    y = torch.tensor([5.0])
    censored = torch.tensor([True])
    log_sigma = torch.tensor(math.log(0.1))
    loss = tobit_censored_loss(pred, y, censored, log_sigma, FLOOR)
    assert loss.item() < 0.01


def test_tobit_loss_claiming_binding_at_nonbinder_costs_more_than_confident_miss():
    """Predicting binding (pKd above floor) at a censored (non-binder) row
    should cost noticeably more than predicting comfortably below the floor."""
    y = torch.tensor([5.0, 5.0])
    censored = torch.tensor([True, True])
    log_sigma = torch.tensor(math.log(0.5))

    confident_nonbinder = torch.tensor([2.0, 2.0])
    claims_binding = torch.tensor([8.0, 8.0])

    loss_good = tobit_censored_loss(confident_nonbinder, y, censored, log_sigma, FLOOR)
    loss_bad = tobit_censored_loss(claims_binding, y, censored, log_sigma, FLOOR)
    assert loss_bad.item() > loss_good.item()


def test_tobit_loss_all_observed_reduces_to_gaussian_nll():
    pred = torch.tensor([5.5, 6.5, 7.5])
    y = torch.tensor([5.0, 7.0, 7.0])
    censored = torch.zeros(3, dtype=torch.bool)
    log_sigma = torch.tensor(math.log(1.0))  # sigma = 1
    loss = tobit_censored_loss(pred, y, censored, log_sigma, FLOOR)

    # With sigma=1, NLL = 0.5*resid^2 + 0.5*log(2*pi) per point (log(sigma)=0)
    resid = y - pred
    expected = (0.5 * resid ** 2 + 0.5 * math.log(2 * math.pi)).mean()
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-5)


def test_learnable_sigma_stays_positive_under_gradient_steps():
    module = LearnableSigma(init_sigma=1.0)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.5)

    pred = torch.tensor([5.5, 6.5, 7.5], requires_grad=True)
    y = torch.tensor([5.0, 7.0, 7.0])
    censored = torch.zeros(3, dtype=torch.bool)

    for _ in range(20):
        optimizer.zero_grad()
        loss = module(pred, y, censored, floor=FLOOR)
        loss.backward()
        optimizer.step()

    assert module.sigma > 0.0
    assert math.isfinite(module.sigma)


def test_tobit_and_hinge_agree_in_direction_of_penalty():
    """Sanity cross-check: both losses should penalize claiming binding at a
    non-binder more than predicting weak binding at a non-binder, even
    though their exact shapes differ."""
    y = torch.tensor([5.0])
    censored = torch.tensor([True])

    weak_pred = torch.tensor([3.0])
    strong_pred = torch.tensor([9.0])

    hinge_weak = censored_hinge_loss(weak_pred, y, censored, floor=FLOOR)
    hinge_strong = censored_hinge_loss(strong_pred, y, censored, floor=FLOOR)
    assert hinge_strong.item() > hinge_weak.item()

    log_sigma = torch.tensor(math.log(1.0))
    tobit_weak = tobit_censored_loss(weak_pred, y, censored, log_sigma, FLOOR)
    tobit_strong = tobit_censored_loss(strong_pred, y, censored, log_sigma, FLOOR)
    assert tobit_strong.item() > tobit_weak.item()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


def test_tobit_gradient_alive_for_confident_false_positive():
    """Regression test: the old clamp(Phi, eps) gave exactly zero gradient here."""
    for p0 in (9.0, 12.0, 30.0):
        p = torch.tensor([p0], requires_grad=True)
        tobit_censored_loss(p, torch.tensor([5.0]), torch.tensor([True]),
                            torch.tensor(-0.7), floor=5.0).backward()
        assert torch.isfinite(p.grad).all()
        assert p.grad.item() > 0.1, f"dead/negative gradient at pred={p0}: {p.grad.item()}"
