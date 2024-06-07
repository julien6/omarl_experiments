from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class algorithm_configuration:
    library: Literal['SB3', 'RLlib'] = field(default='SB3')
    cpu_core_number: Union[int, Literal['auto']] = 'auto'
    gpu_core_number: Union[int, Literal['auto']] = 'auto'

@dataclass
class PPO_configuration:
    policy: str | type[ActorCriticPolicy],
    learning_rate: float | Schedule = 0.0003,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float | Schedule = 0.2,
    clip_range_vf: float | Schedule | None = None,
    normalize_advantage: bool = True,
    ent_coef: float = 0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_sde: bool = False

class SB3_configuration(algorithm_configuration):
    pass


class RLlib_configuration(algorithm_configuration):
    pass


class library_configuration_factory:

    def __init__(self) -> None:
        pass

    def PPO() -> algorithm_configuration:
        return

    def MADDPG() -> algorithm_configuration:
        return


class SB3_configuration_factory(library_configuration_factory):

    def __init__(self) -> None:
        pass

    def PPO() -> SB3_configuration:
        pass

    def MADDPG() -> algorithm_configuration:
        return


class RLlib_configuration_factory(library_configuration_factory):

    def __init__(self) -> None:
        super().__init__()

    def PPO() -> algorithm_configuration:
        pass

    def MADDPG() -> algorithm_configuration:
        return


class prahom_algorithm_factory:

    def __init__(self) -> None:
        pass

    def SB3(self) -> SB3_configuration_factory:
        return SB3_configuration_factory

    def RLlib(self) -> RLlib_configuration_factory:
        return SB3_configuration_factory


prahom_alg_fac = prahom_algorithm_factory()

if __name__ == '__main__':

    prahom_alg_fac.RLlib().PPO()
