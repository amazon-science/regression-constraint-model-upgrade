import itertools


def get_hyperparameters(hyperparameters, add_hparam=None):
    if isinstance(hyperparameters, dict):
        hyperparameters = [[(hp_name, value) for value in hyperparameters[hp_name]] for hp_name in hyperparameters]
        hyperparameters = list(itertools.product(*hyperparameters))
        hyperparameters = [{k: v for k, v in hparam} for hparam in hyperparameters]
    elif not isinstance(hyperparameters, list):
        raise ValueError()

    def _get_hparam_str(hparam):
        return '_'.join([k + str(v) for k, v in sorted(hparam.items())])

    for hparam in hyperparameters:
        if add_hparam is not None:
            hparam.update(add_hparam)

    hparam_strs = [_get_hparam_str(hparam) for hparam in hyperparameters]
    assert len(hyperparameters) == len(set(hparam_strs))

    return list(zip(hyperparameters, hparam_strs))
