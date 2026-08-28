import numpy as np


def coin_minus_labor(
        coin_endowment, total_labor, labor_coefficient
):
    # https://en.wikipedia.org/wiki/Isoelastic_utility

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = coin_endowment - util_l

    return util


def isoelastic_coin_minus_labor(
        coin_endowment, total_labor, isoelastic_eta, labor_coefficient
):

    # https://en.wikipedia.org/wiki/Isoelastic_utility
    assert 0 <= isoelastic_eta <= 1.0

    # Utility from coin endowment
    if isoelastic_eta == 1.0:  # dangerous
        util_c = np.log(np.maximum(1, coin_endowment))
    else:  # isoelastic_eta >= 0
        if np.all(coin_endowment >= 0):
            util_c = (coin_endowment ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta)
        else:
            util_c = coin_endowment - 1

    # disutility from labor
    util_l = total_labor * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util


def _planner_welfare(coin_endowments):
    """Social welfare term of the planner objective, by agent count.

    Single agent: welfare is that firm's profit. Equality is undefined for
    one agent (the Gini normaliser ((n-1)/n) is zero) and productivity/n
    collapses to the same number, so the richer form buys nothing here.

    Multi agent: the equality * productivity objective, which is what
    actually distinguishes a cap-and-trade planner from a profit maximiser.
    This branch used to be missing -- the planner scored
    world.agents[0] alone, so with n_agents=5 it optimised one firm's
    profit and was blind to the other four.
    """
    coin_endowments = np.asarray(coin_endowments, dtype=float)
    n_agents = len(coin_endowments)

    if n_agents == 1:
        return float(coin_endowments[0])

    return get_equality(coin_endowments) * (get_productivity(coin_endowments) / n_agents)


def planner_strategy(coin_endowments, mobile_idx, remained_idx, mobile_coefficient):
    """remained idx is the indext that is still left for the planner to allocate"""

    idx_used_mobile = np.exp(sum([-1 * mobile_coefficient * idx for idx in
                                  mobile_idx]))  # if agents spend more than allocated index, this term decreases to <1 other >1
    idx_overspent = min(0, remained_idx)  # Penalty for overspending index by the planner

    util = _planner_welfare(coin_endowments) * idx_used_mobile - 50.0 * idx_overspent ** 2
    return util

def get_gini(endowments):

    n_agents = len(endowments)

    # ((n-1)/n) is 0 for a single agent, which turned the division below
    # into a silent 0/0 NaN. There is no dispersion to measure with one
    # agent, so equality is 1.
    if n_agents < 2:
        return 0.0

    if n_agents < 30:  # Slower. Accurate for all n.
        diff_ij = np.abs(
            endowments.reshape((n_agents, 1)) - endowments.reshape((1, n_agents))
        )
        diff = np.sum(diff_ij)
        norm = 2 * n_agents * endowments.sum(axis=0)
        unscaled_gini = diff / (norm + 1e-10)
        gini = unscaled_gini / ((n_agents - 1) / n_agents)
        return gini

    # Much faster. Slightly overestimated for low n.
    s_endows = np.sort(endowments)
    return 1 - (2 / (n_agents + 1)) * np.sum(
        np.cumsum(s_endows) / (np.sum(s_endows) + 1e-10)
    )


def get_equality(endowments):

    return 1 - get_gini(endowments)


def get_productivity(coin_endowments):

    return np.sum(coin_endowments)

def planner_metrics(coin_endowments, mobile_idx, remained_idx, mobile_coefficient):
    coin_endowments = np.asarray(coin_endowments, dtype=float)
    n_agents = len(coin_endowments)

    idx_used_mobile = np.exp(sum([-1 * mobile_coefficient * idx for idx in
                                  mobile_idx]))  # if agents spend more than allocated index, this term decreases to <1 other >1
    idx_overspent = min(0, remained_idx)  # Penalty for overspending index by the planner

    util = _planner_welfare(coin_endowments) * idx_used_mobile - 50.0 * idx_overspent ** 2

    planner_metrix = {
        "util": util,
        "prod": get_productivity(coin_endowments) / n_agents,
        "equality": get_equality(coin_endowments),
        "mobile_idx_used": mobile_idx
    }
    return planner_metrix