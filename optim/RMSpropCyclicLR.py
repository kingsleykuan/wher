from torch import optim

# TODO: Check saving and restoring of optimizer / learning rate scheduler
class RMSpropCyclicLR(optim.RMSprop):
    """
    RMSprop optimizer with CyclicLR learning rate scheduler.
    """
    def __init__(self,
            params,
            lr=1e-7,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False,
            base_lr=1e-7,
            max_lr=1e-1,
            step_size_up=2000,
            step_size_down=None,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle',
            cycle_momentum=False,
            base_momentum=0.8,
            max_momentum=0.9,
            last_epoch=-1):

        super().__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered)

        if mode == 'exp_range':
            scale_fn = lambda x: gamma**(x)
            scale_mode = 'cycle'

        self.cyclic_lr = optim.lr_scheduler.CyclicLR(
            self,
            base_lr,
            max_lr,
            step_size_up=step_size_up,
            step_size_down=step_size_down,
            mode=mode,
            gamma=gamma,
            scale_fn=scale_fn,
            scale_mode=scale_mode,
            cycle_momentum=cycle_momentum,
            base_momentum=base_momentum,
            max_momentum=max_momentum,
            last_epoch=last_epoch)

    def step(self, closure=None):
        o = super().step(closure)
        self.cyclic_lr.step()
        return o
