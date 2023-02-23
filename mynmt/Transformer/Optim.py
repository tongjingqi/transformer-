class ScheduledOptim:
    def __init__(self, optimizer, lr_mul=2.0, d_model=512, n_warmup_steps=4000):
        """热启动->降低"""
        self.optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        """Step with the inner optimizer"""
        self._update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients with the inner optimizer"""
        self.optimizer.zero_grad()

    def _get_lr_scale(self):
        return (self.d_model ** -0.5) * min(self.n_steps ** (-0.5),
                                            self.n_steps * self.n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
