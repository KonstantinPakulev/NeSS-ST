import torch

from abc import ABC, abstractmethod
from torch.cuda import Event

import source.core.namespace as ns

from source.core.model import get_num_branches

MEASURE_TIME = 'measure_time'


class ModuleWrapper(ABC):

    def __init__(self, experiment_config):
        self.measure_time = experiment_config.get(MEASURE_TIME, False)

    def __call__(self, engine, device, batch, inference_bundle, endpoint):
        item_keys = list(batch.keys()) + list(inference_bundle.keys()) + list(endpoint.keys())

        if self.measure_time:
            inference_time = 0

        num_branches = get_num_branches(self._get_forward_base_key(), item_keys) + 1

        for i in range(1, num_branches):
            if self.measure_time:
                start = Event(enable_timing=True)
                end = Event(enable_timing=True)

                start.record()
                self._forward_branch(engine, device, i, batch, inference_bundle, endpoint)
                end.record()

                torch.cuda.synchronize()

                inference_time += start.elapsed_time(end)

            else:
                self._forward_branch(engine, device, i, batch, inference_bundle, endpoint)

        if self.measure_time:
            endpoint[f"{self.__class__.__name__}_{ns.INFERENCE_TIME}"] = inference_time / num_branches

    @abstractmethod
    def _get_forward_base_key(self):
        ...

    @abstractmethod
    def _forward_branch(self, engine, device, i, batch, inference_bundle, endpoint):
        ...

    def process(self, engine, device, batch, inference_bundle, endpoint, eval_params):
        item_keys = list(batch.keys()) + list(inference_bundle.keys()) + list(endpoint.keys())

        if self._get_process_base_key() is None:
            num_branches = get_num_branches(self._get_forward_base_key(), item_keys)

        else:
            num_branches = get_num_branches(self._get_process_base_key(), item_keys)

        if self.measure_time:
            process_time = 0

        for i in range(1, num_branches + 1):
            if self.measure_time:
                start = Event(enable_timing=True)
                end = Event(enable_timing=True)

                start.record()
                self._process_branch(engine, device, i, batch, inference_bundle, endpoint, eval_params)
                end.record()

                torch.cuda.synchronize()

                process_time += start.elapsed_time(end)

            else:
                self._process_branch(engine, device, i, batch, inference_bundle, endpoint, eval_params)

        if self.measure_time:
            endpoint[f"{self.__class__.__name__}_{ns.PROCESS_TIME}"] = process_time / num_branches

    @abstractmethod
    def _get_process_base_key(self):
        ...

    @abstractmethod
    def _process_branch(self, engine, device, i, batch, inference_bundle, endpoint, eval_params):
        ...

    @abstractmethod
    def get(self):
        ...


def get_ith_key_input(base_key, i, batch, bundle, endpoint, device):
    keyi = f"{base_key}{i}"

    if keyi in batch:
        inputi = batch[keyi].to(device)

    elif bundle is not None and keyi in bundle:
        inputi = bundle[keyi]

    else:
        inputi = endpoint[keyi]

    return inputi
