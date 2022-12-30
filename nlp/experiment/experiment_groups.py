from methods.main import get_run

hp = {'MASSIVE': dict(e=[3, 6, 10], lr=[3e-5, 6e-5]),
      'ag_news': dict(e=[2, 3, 6], lr=[3e-5, 6e-5]),
      'banking77': dict(e=[3, 6, 10], lr=[3e-5, 6e-5])}

hp_pre = {'MASSIVE': dict(e=[16], lr=[6e-5]),
          'ag_news': dict(e=[8], lr=[6e-5]),
          'banking77': dict(e=[16], lr=[6e-5])}

exp_groups = dict()

# Save for BCWI
exp_groups['new_model_add_data'] = [
    get_run(hp=dict(MASSIVE=dict(e=[6], lr=[6e-05]),
                    banking77=dict(e=[10], lr=[3e-05]),
                    ag_news=dict(e=[3], lr=[3e-05])),
            from_model='old',
            save_model=True,
            cname='bestACC')
]
exp_groups['target_model_add_data'] = [
    get_run(hp=dict(MASSIVE=dict(e=[16], lr=[6e-05]),
                    banking77=dict(e=[16], lr=[6e-05]),
                    ag_news=dict(e=[8], lr=[6e-05])),
            from_model='pretrained',
            save_model=False,
            cname='bestACC')
]

exp_groups['new_model_add_classes'] = [
    get_run(hp=dict(MASSIVE=dict(e=[10], lr=[3e-05]),
                    banking77=dict(e=[10], lr=[6e-05]),
                    ag_news=dict(e=[6], lr=[3e-05])),
            from_model='old',
            save_model=True,
            cname='bestACC'),
]
exp_groups['target_model_add_classes'] = [
    get_run(hp=dict(MASSIVE=dict(e=[16], lr=[6e-05]),
                    banking77=dict(e=[16], lr=[6e-05]),
                    ag_news=dict(e=[8], lr=[6e-05])),
            from_model='pretrained',
            save_model=False,
            cname='bestACC')
]


# Parameter Efficient Finetuning
exp_groups['bitfit_add_data'] = [
    get_run(dict(e=[7, 12, 16], lr=[1e-4, 1e-3, 1e-2]), ft_layers=['bias'])
]
exp_groups['bitfit_add_classes'] = [
    get_run(dict(e=[7, 12, 16], lr=[1e-4, 1e-3, 1e-2]), ft_layers=['bias', 'classifier'])
]
exp_groups['ia3'] = [
    get_run(dict(e=[8, 12, 16], lr=[1e-4, 1e-3, 1e-2]), from_model='old', ia3=dict(variant='base')),
]


# Mixout
exp_groups['mixout'] = [
    get_run(hp, mixout=0.05),
    get_run(hp, mixout=0.1),
    get_run(hp, mixout=0.2),
    get_run(hp, mixout=0.3),
    get_run(hp, mixout=0.4),
    get_run(hp, mixout=0.5, log_batch=True),
    get_run(hp, mixout=0.6),
    get_run(hp, mixout=0.7),
    get_run(hp, mixout=0.8),
    get_run(hp, mixout=0.9),
    get_run(hp, mixout=0.95),
    get_run(hp, mixout=0.99),
    get_run(hp, mixout=0.999)
]

# Prior Weight Decay
exp_groups['pre_wd'] = [
    get_run(hp, prior_wd=0.01),
    get_run(hp, prior_wd=0.1),
    get_run(hp, prior_wd=1.0),
    get_run(hp, prior_wd=10.0),
    get_run(hp, prior_wd=100.0),
    get_run(hp, prior_wd=200.0),
    get_run(hp, prior_wd=1000.0),
    get_run(hp, prior_wd=2000.0),
    get_run(hp, prior_wd=4000.0),
    get_run(hp, prior_wd=10000.0),
    get_run(hp, prior_wd=100000.0),
]

# EWC
exp_groups['ewc'] = [
    get_run(hp, ewc=0.0000001),
    get_run(hp, ewc=0.000001),
    get_run(hp, ewc=0.00001),
    get_run(hp, ewc=0.0001),
    get_run(hp, ewc=0.001),
    get_run(hp, ewc=0.01),
    get_run(hp, ewc=0.1),
    get_run(hp, ewc=1.0),
    get_run(hp, ewc=2.0),
    get_run(hp, ewc=5.0),
    get_run(hp, ewc=10.0),
    get_run(hp, ewc=50.0),
    get_run(hp, ewc=100.0),
    get_run(hp, ewc=1000.0),
    get_run(hp, ewc=10000.0)
]


# Distillation
exp_groups['distillation'] = [
    get_run(hp, from_model='old', distill=dict(variant='oldOnlyA', alpha=1.0, temperature=2.0)),
]

# Soup
exp_groups['ensemble_add_data'] = [
    get_run(hp=dict(MASSIVE=dict(e=[6], lr=[6e-5], seed=list(range(16))),
                    banking77=dict(e=[10], lr=[3e-5], seed=list(range(16))),
                    ag_news=dict(e=[3], lr=[3e-05], seed=list(range(16)))),
            from_model='old',
            save_model=True,
            cname='bestACC_ensemble')
]

exp_groups['ensemble_add_classes'] = [
    get_run(hp=dict(MASSIVE=dict(e=[10], lr=[3e-5], seed=list(range(16))),
                    banking77=dict(e=[10], lr=[6e-5], seed=list(range(16))),
                    ag_news=dict(e=[6], lr=[3e-05], seed=list(range(16)))),
            from_model='old',
            save_model=True,
            cname='bestACC_ensemble')
]
