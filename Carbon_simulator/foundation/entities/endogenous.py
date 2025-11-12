from Carbon_simulator.foundation.base.registrar import Registry


class Endogenous:
    """Base class for endogenous entity classes.

    Endogenous entities are those that, conceptually, describe the internal state
    of an agent. This provides a convenient way to separate physical entities (which
    may exist in the world, be exchanged among agents, or are otherwise in principal
    observable by others) from endogenous entities (such as the amount of labor
    effort an agent has experienced).

    Endogenous entities are registered in the "endogenous" portion of an agent's
    state and should only be observable by the agent itself.
    """

    name = None

    def __init__(self):
        assert self.name is not None


endogenous_registry = Registry(Endogenous)


@endogenous_registry.add
class Labor(Endogenous):
    """Labor accumulated through working. Included in all environments by default."""

    name = "Labor"


@endogenous_registry.add
class LaborCost(Endogenous):
    """Labor accumulated through working. Included in all environments by default."""

    name = "LaborCost"


@endogenous_registry.add
class Carbon_emission(Endogenous):
    """Labor accumulated through working. Included in all environments by default."""

    name = "Carbon_emission"


@endogenous_registry.add
class Costs(Endogenous):
    """Costs associated with various activities, such as manufacturing or research."""

    name = "Costs"


@endogenous_registry.add
class Revenue(Endogenous):
    """Profit generated from activities such as manufacturing / carbontrade."""

    name = "Revenue"


@endogenous_registry.add
class Reward(Endogenous):
    """reward for every agent """
    name = "Reward"


@endogenous_registry.add
class RewardPlanner(Endogenous):
    """reward for every agent """
    name = "RewardPlanner"


@endogenous_registry.add
class Startidx(Endogenous):
    """reward for every agent """
    name = "Startidx"


@endogenous_registry.add
class PastUtility(Endogenous):
    """reward for every agent """
    name = "PastUtility"


@endogenous_registry.add
class LaborUtility(Endogenous):
    """reward for every agent """
    name = "LaborUtility"


@endogenous_registry.add
class CoinUtility(Endogenous):
    """reward for every agent """
    name = "CoinUtility"


@endogenous_registry.add
class CurrentUtility(Endogenous):
    """reward for every agent """
    name = "CurrentUtility"


@endogenous_registry.add
class CoinEndowment(Endogenous):
    """reward for every agent """
    name = "CoinEndowment"
