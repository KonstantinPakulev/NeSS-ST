import torch

from ignite.engine import Engine

import source.core.namespace as ns


"""
Loop modes
"""


class Loop:

    def __init__(self, device, loop_mode,
                 model_mode_wrapper, criterion_chain, optimizer_wrapper,
                 dataset, loader):

        def iteration(engine, batch):
            engine.state.output = None

            self._load_model()

            endpoint = self._loop_iteration(engine, batch)

            endpoint2cpu(endpoint)

            return endpoint

        engine = Engine(iteration)

        model_mode_wrapper.attach(engine, device)

        if criterion_chain is not None:
            criterion_chain.attach(engine)

        if hasattr(dataset, 'attach'):
            dataset.attach(engine)

        self.engine = engine

        self.device = device
        self.loop_mode = loop_mode

        self.model_mode_wrapper = model_mode_wrapper
        self.criterion_chain = criterion_chain
        self.is_loaded = False

        self.dataset = dataset
        self.loader = loader
        self.optimizer_wrapper = optimizer_wrapper

    def run(self, num_epochs, return_output=False):
        self.engine.state.return_output = return_output
        self.engine.run(self.loader, max_epochs=num_epochs)

        if return_output:
            return self.engine.state.batch, self.engine.state.output, self.engine.state.metrics
        else:
            return None

    def unload_model(self):
        if self.model_mode_wrapper.has_model():
            self.model_mode_wrapper.cpu()
            self.is_loaded = False

    def has_optimizer(self):
        return self.optimizer_wrapper is not None

    def _load_model(self):
        if self.model_mode_wrapper.has_model():
            if self.loop_mode == ns.TRAIN:
                self.model_mode_wrapper.train()

            elif self.loop_mode == ns.EVAL:
                self.model_mode_wrapper.eval()

            else:
                raise Exception(f"Unknown mode {self.loop_mode}")

            self.model_mode_wrapper.to(self.device)

            if self.has_optimizer():
                self.optimizer_wrapper.to(self.device)

            self.is_loaded = True

    def _loop_iteration(self, engine, batch):
        inference_bundle = {}
        endpoint = {}

        if self.loop_mode == ns.TRAIN:
            self.model_mode_wrapper(engine, self.device, batch, inference_bundle, endpoint)
            self.criterion_chain(engine, self.device, batch, inference_bundle, endpoint)
            self.optimizer_wrapper(engine, inference_bundle)

            detach_inference_bundle_and_endpoint(inference_bundle, endpoint)

        elif self.loop_mode == ns.EVAL:
            with torch.no_grad():
                if self.model_mode_wrapper.has_model():
                    self.model_mode_wrapper(engine, self.device, batch, inference_bundle, endpoint)

                if self.criterion_chain is not None:
                    self.criterion_chain(engine, self.device, batch, inference_bundle, endpoint)

        else:
            raise Exception(f"Unknown mode {loop_mode}")

        with torch.no_grad():
            if self.model_mode_wrapper.do_process():
                self.model_mode_wrapper.process(engine, self.device, batch, inference_bundle, endpoint)

        if self.engine.state.return_output:
            copy_dict2dict(inference_bundle, endpoint)

        return endpoint


"""
Support utils
"""


def detach_inference_bundle_and_endpoint(inference_bundle, endpoint):
    for k, v in inference_bundle.items():
        inference_bundle[k] = v.detach()

    for k, v in endpoint.items():
        endpoint[k] = v.detach()

    return inference_bundle, endpoint

def copy_dict2dict(d1, d2):
    for k, v in d1.items():
        d2[k] = v

def endpoint2cpu(endpoint):
    for k, v in endpoint.items():
        if torch.is_tensor(v):
            endpoint[k] = v.cpu()

"""
Legacy code
"""


# def prepare_bundle_and_endpoint(bundle, endpoint):
#     for k, v in bundle.items():
#         bundle[k] = v.detach()
#
#     for k, v in endpoint.items():
#         endpoint[k] = v.detach()

# def finalize_endpoint(endpoint):
#     for k, v, in endpoint.items():
#         endpoint[k] = v.cpu()

# def check_requires_grad(endpoint):
#     for v in endpoint.values():
#         if torch.is_tensor(v) and v.requires_grad:
#             raise AssertionError('Endpoint can not contain tensors where requires_grad=True')

#         and not self.model_mode_wrapper.is_on_device(self.device)

# def prepare_endpoint(endpoint):
#     for k, v, in endpoint.items():
#         endpoint[k] = v.cpu()
