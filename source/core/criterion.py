from abc import ABC, abstractmethod

import torch

import source.core.namespace as ns

from source.core.evaluation import TransformerHandler, KeyTransformer


class CriterionWrapper(ABC):

    def __call__(self, engine, device, batch, bundle, endpoint):
        return self.forward(engine, device, batch, bundle, endpoint)

    @abstractmethod
    def forward(self, engine, device, batch, bundle, endpoint):
        ...

    @abstractmethod
    def get(self):
        ...


class CriterionChain:

    def __init__(self, criteria_wrappers, eval_config):
        self.criteria_wrappers = criteria_wrappers
        self.eval_config = eval_config

    def __call__(self, engine, device, batch, bundle, endpoint):
        losses = []

        for c_w in self.criteria_wrappers:
            loss = c_w(engine, device, batch, bundle, endpoint)

            if loss is not None:
                losses.append(loss)

        if len(losses) != 0:
            bundle[ns.LOSS] = torch.stack(losses).sum()

    def attach(self, engine):
        if self.eval_config is not None:
            loss_log_iter = self.eval_config.get(ns.LOSS_LOG_ITER)

            if loss_log_iter is not None:
                losses_names = []

                for c_w in self.criteria_wrappers:
                    losses_namesi = c_w.get()

                    if losses_namesi is not None:
                        losses_names.extend(losses_namesi)

                if len(losses_names) != 0:
                    TransformerHandler(KeyTransformer(losses_names), loss_log_iter).attach(engine, losses_names)
