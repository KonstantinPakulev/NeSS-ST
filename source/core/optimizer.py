import torch
from torch.optim import Adam

import source.core.namespace as ns


class OptimizerWrapper:

    def __init__(self, parameters, optimizer_config):
        self.step = optimizer_config.step

        self.optimizer = Adam(parameters, optimizer_config.lr)

    def __call__(self, engine, bundle):
        loss = bundle[ns.LOSS]

        self.optimizer.zero_grad()
        loss.backward()

        if self.step:
            self.optimizer.step()

    def to(self, device):
        for v in self.optimizer.state.values():
            if isinstance(v, torch.Tensor):
                v.data = v.data.to(device)
                if v._grad is not None:
                    v._grad.data = v._grad.data.to(device)

            elif isinstance(v, dict):
                for e_v in v.values():
                    if isinstance(e_v, torch.Tensor):
                        e_v.data = e_v.data.to(device)
                        if e_v._grad is not None:
                            e_v._grad.data = e_v._grad.data.to(device)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)
