class algorithm_configuration:
    pass


class SB3_configuration(algorithm_configuration):
    pass


class RLlib_configuration(algorithm_configuration):
    pass


class library_configuration_factory:

    def __init__(self) -> None:
        pass

    def default_PPO() -> algorithm_configuration:
        return

    def default_MADDPG() -> algorithm_configuration:
        return


class SB3_configuration_factory(library_configuration_factory):

    def __init__(self) -> None:
        pass

    def default_PPO() -> SB3_configuration:
        pass

    def default_MADDPG() -> algorithm_configuration:
        return


class RLlib_configuration_factory(library_configuration_factory):

    def __init__(self) -> None:
        super().__init__()

    def default_PPO() -> algorithm_configuration:
        pass

    def default_MADDPG() -> algorithm_configuration:
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

    prahom_alg_fac.RLlib().default_PPO()
