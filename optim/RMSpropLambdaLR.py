from torch import optim

# TODO: Check saving and restoring of optimizer / learning rate scheduler
class RMSpropLambdaLR(optim.RMSprop):
    """
    RMSprop optimizer with LambdaLR learning rate scheduler.

    Sets the learning rate to the initial lr times a given function.
    """
    def __init__(self,
            params,
            lr=1e-7,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False,
            lr_lambda=lambda x: x,
            last_epoch=-1):

        super().__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered)

        self.cyclic_lr = optim.lr_scheduler.LambdaLR(
            self,
            lr_lambda,
            last_epoch=last_epoch)

    def step(self, closure=None):
        x = super().step(closure)
        self.cyclic_lr.step()
        return x
