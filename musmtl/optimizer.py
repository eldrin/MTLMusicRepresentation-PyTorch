class MultipleOptimizerList:
    """ Simple wrapper for the multiple optimizer scenario """
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        return [v.state_dict() for v in self.optimizers]

    def load_state_dict(self, state):
        for i, v in enumerate(state):
            self.optimizers[i].load_state_dict(v)


class MultipleOptimizerDict:
    """ Simple wrapper for the multiple optimizer scenario """
    def __init__(self, **op):
        self.optimizers = op

    def zero_grad(self):
        for task, op in self.optimizers.items():
            op.zero_grad()

    def step(self, keys):
        for key in keys:
            if key in self.optimizers:
                self.optimizers[key].step()

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.optimizers.items()}

    def load_state_dict(self, state):
        for k, v in state.items():
            self.optimizers[k].load_state_dict(v)
