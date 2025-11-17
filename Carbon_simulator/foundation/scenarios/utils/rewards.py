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
        util_c = np.log(np.max(1, coin_endowment))
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


def planner_strategy(profit, mobile_idx, remained_idx, mobile_coefficient):
    """remained idx is the indext that is still left for the planner to allocate"""

    idx_used_mobile = np.exp(sum([-1 * mobile_coefficient * idx for idx in
                                  mobile_idx]))  # if agents spend more than allocated index, this term decreases to <1 other >1
    idx_overspent = min(0, remained_idx)  # Penalty for overspending index by the planner

    util = profit * idx_used_mobile - 50.0 * idx_overspent ** 2
    return util

def get_gini(endowments):

    n_agents = len(endowments)

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

def planner_metrics(profit, mobile_idx, remained_idx, mobile_coefficient):
    idx_used_mobile = np.exp(sum([-1 * mobile_coefficient * idx for idx in
                                  mobile_idx]))  # if agents spend more than allocated index, this term decreases to <1 other >1
    idx_overspent = min(0, remained_idx)  # Penalty for overspending index by the planner

    util = profit * idx_used_mobile - 50.0 * idx_overspent ** 2

    planner_metrix = {
        "util": util,
        "prod": profit,
        "mobile_idx_used": mobile_idx
    }
    return planner_metrix