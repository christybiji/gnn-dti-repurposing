"""
Loss functions for censored drug-target affinity regression.

This module contains two losses:

  - `censored_hinge_loss`: a reimplementation of the notebook's
    `censored_mse_loss` (Section 10). Ordinary squared error on observed
    pairs; a one-sided quadratic hinge on censored pairs.

  - `tobit_censored_loss`: the proper Tobit (censored-normal) negative
    log-likelihood that the notebook names as "the more principled choice"
    but explicitly does not implement. It treats a censored observation
    ("Kd is at least the assay floor") using the correct Gaussian tail
    probability instead of a hand-shaped hinge, and it *learns* the noise
    scale (sigma) jointly with the rest of the model rather than assuming
    a fixed penalty shape.

Both losses share the same call signature so either can be dropped into
the notebook's Section 13 training loop with a one-line change:

    loss = censored_mse_loss(out, batch.y, batch.cens)      # before
    loss = tobit_censored_loss(out, batch.y, batch.cens, log_sigma)  # after

See README.md for the exact wiring instructions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def censored_hinge_loss(pred: torch.Tensor, y: torch.Tensor, censored: torch.Tensor,
                         floor: float) -> torch.Tensor:
    """MSE on observed pairs; one-sided quadratic hinge on censored pairs.

    Equivalent in behavior to the notebook's `censored_mse_loss` (Section 10):
    a censored label means "Kd is at least this weak", so a prediction that is
    *weaker* than the floor costs nothing, and only predictions that claim
    binding stronger than the floor are penalized, quadratically.

    Parameters
    ----------
    pred : (N,) predicted pKd
    y : (N,) true pKd (only meaningful for observed rows; for censored rows
        this is conventionally the floor itself, as in the notebook)
    censored : (N,) bool mask, True where the row is right-censored
    floor : the pKd floor (e.g. PKD_FLOOR in the notebook: 9 - log10(10_000))
    """
    censored = censored.bool()
    observed = ~censored

    total = pred.new_zeros(())
    if observed.any():
        total = total + ((pred[observed] - y[observed]) ** 2).sum()
    if censored.any():
        total = total + (torch.clamp(pred[censored] - floor, min=0.0) ** 2).sum()
    return total / pred.numel()


def tobit_censored_loss(pred: torch.Tensor, y: torch.Tensor, censored: torch.Tensor,
                         log_sigma: torch.Tensor, floor: float,
                         eps: float = 1e-6) -> torch.Tensor:
    """Tobit (censored-normal) negative log-likelihood.

    Model: the true pKd is Normal(pred, sigma^2). For an observed row we
    know the exact value and use the ordinary Gaussian NLL. For a censored
    row we only know the true value is <= floor (a right-censored *Kd*, i.e.
    the pair does not bind, corresponds to a value at or below the pKd
    floor), so the likelihood contribution is the probability mass the
    model places below the floor: Phi((floor - pred) / sigma).

    This replaces the hinge's fixed quadratic shape with the actual
    likelihood implied by treating residuals as Gaussian, and it estimates
    sigma from the data instead of assuming one. `log_sigma` should be a
    learnable scalar (or per-group) parameter, optimized jointly with the
    rest of the network.

    Parameters
    ----------
    pred : (N,) predicted pKd (interpreted as the mean of the Normal)
    y : (N,) true pKd for observed rows (ignored for censored rows)
    censored : (N,) bool mask, True where the row is right-censored
    log_sigma : scalar tensor, log of the Normal's noise scale. Kept in log
        space so optimization is unconstrained while sigma stays positive.
    floor : the pKd floor (censoring threshold)
    eps : added to sigma for numerical safety

    Returns
    -------
    Scalar mean negative log-likelihood over the batch.
    """
    censored = censored.bool()
    observed = ~censored
    sigma = torch.exp(log_sigma) + eps

    n = pred.numel()
    total = pred.new_zeros(())

    if observed.any():
        resid = (y[observed] - pred[observed]) / sigma
        nll_obs = 0.5 * resid ** 2 + torch.log(sigma) + _LOG_SQRT_2PI
        total = total + nll_obs.sum()

    if censored.any():
        # log Phi(z) via log_ndtr: stable in the far tail. Do NOT clamp Phi(z)
        # before the log -- that zeroes the gradient for confident false
        # positives (pred far above the floor), the rows that most need it.
        z = (floor - pred[censored]) / sigma
        total = total - torch.special.log_ndtr(z).sum()

    return total / n


class LearnableSigma(nn.Module):
    """Convenience wrapper holding the Tobit loss's learnable noise scale.

    Usage in the training loop:

        tobit = LearnableSigma()
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(tobit.parameters()), lr=cfg.lr
        )
        ...
        loss = tobit(pred, batch.y, batch.cens, floor=PKD_FLOOR)
    """

    def __init__(self, init_sigma: float = 1.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))

    def forward(self, pred: torch.Tensor, y: torch.Tensor, censored: torch.Tensor,
                floor: float) -> torch.Tensor:
        return tobit_censored_loss(pred, y, censored, self.log_sigma, floor)

    @property
    def sigma(self) -> float:
        return float(torch.exp(self.log_sigma.detach()))
