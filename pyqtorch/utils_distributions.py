from __future__ import annotations

import math
from collections import Counter

import torch
from torch import Tensor


def shannon_entropy(counter: Counter) -> float:
    return float(-sum([count * math.log(count) for count in counter.values()]))


def js_divergence_counters(counter_p: Counter, counter_q: Counter) -> float:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    represented as Counter objects.
    The JSD is calculated using only the shared keys between the two input Counter objects.

    Args:
        counter_p (Counter): Counter of bitstring counts for probability mass function P.
        counter_q (Counter): Counter of bitstring counts for probability mass function Q.

    Returns:
        float: The Jensen-Shannon divergence between counter_p and counter_q.
    """
    # Normalise counters
    normalisation_p = sum([count for count in counter_p.values()])
    normalisation_q = sum([count for count in counter_q.values()])
    counter_p = Counter({k: v / normalisation_p for k, v in counter_p.items()})
    counter_q = Counter({k: v / normalisation_q for k, v in counter_q.items()})

    average_proba_counter = counter_p + counter_q
    average_proba_counter = Counter(
        {k: v / 2.0 for k, v in average_proba_counter.items()}
    )
    average_entropy = shannon_entropy(average_proba_counter)

    entropy_p = shannon_entropy(counter_p)
    entropy_q = shannon_entropy(counter_q)
    return float(average_entropy - (entropy_p + entropy_q) / 2.0)


def js_divergence(
    proba_mass_P: Tensor, proba_mass_Q: Tensor, epsilon: float = 1e-6
) -> Tensor:
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Args:
        proba_mass_P (Tensor): Probability mass function P
        proba_mass_Q (Tensor): Probability mass function Q
        epsilon (float, optional): Small number to avoid 0 division. Defaults to 1e-6.

    Returns:
        Tensor: The Jensen-Shannon divergence between P and Q.
    """

    # Clamp values to avoid log(0)
    proba_mass_P = torch.clamp(proba_mass_P, min=epsilon)
    proba_mass_Q = torch.clamp(proba_mass_Q, min=epsilon)

    # Calculate the middle point distribution
    m = 0.5 * (proba_mass_P + proba_mass_Q)

    # Calculate KL divergence for both distributions with respect to m
    kl_p_m = torch.sum(proba_mass_P * torch.log2(proba_mass_P / m), dim=-1)
    kl_q_m = torch.sum(proba_mass_Q * torch.log2(proba_mass_Q / m), dim=-1)

    # Jensen-Shannon divergence is the average of the two KL divergences
    return 0.5 * (kl_p_m + kl_q_m)
