from torch.optim import Optimizer


# Prior Weight Decay (https://arxiv.org/abs/1909.11299) regularization for finetuning.
# Moves the current weights towards the old model weights during each gradient weight update.
# https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py


def validate_argument(pre_wd, name):
    if not pre_wd:
        return pre_wd, name
    name += '_pre_wd' + str(pre_wd)
    return pre_wd, name


def apply(model, pre_wd, config):
    if not pre_wd:
        return model

    config['prior_wd'] = pre_wd
    # Set regular weight decay to 0 when prior weight decay is active
    config['weight_decay'] = 0.0
    print('set wd to 0')
    return model


class PriorWD(Optimizer):
    def __init__(self, optimizer, value, skip_classifier=False):
        super(PriorWD, self).__init__(optimizer.param_groups, optimizer.defaults)
        print('PriorWD active')
        print('skip_classifier', skip_classifier)

        self.param_groups = optimizer.param_groups
        self.optimizer = optimizer
        self.value = value
        self.skip_classifier = skip_classifier

        # Keep a copy of the old model weights in memory
        self.prior_params = {}
        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                self.prior_params[id(p)] = p.detach().clone()

    # Optimizer update step
    def step(self, closure=None, *args, **kwargs):
        for i, group in enumerate(self.param_groups):
            alpha = group["lr"] * self.value
            for j, p in enumerate(group["params"]):
                if self.skip_classifier and j >= len(group["params"])-2:
                    continue
                # Move current weights towards the weights of the old model. Alpha is the size of the step.
                p.data.add_(p.data - self.prior_params[id(p)], alpha=-alpha)

        loss = self.optimizer.step(closure, *args, **kwargs)

        return loss
