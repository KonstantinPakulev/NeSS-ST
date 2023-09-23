import numpy as np

from enum import Enum
from abc import ABC, abstractmethod
from ignite.engine import Events, State
from multiprocessing import Pool
from queue import deque


REDUCE = 'reduce'
OUTPUT_KEYS = 'output_keys'


"""
Transformers
"""


class BaseTransformer(ABC):

    def on_epoch_started(self, engine):
        pass

    def on_iteration_completed(self, engine, batch, endpoint):
        return None

    def on_iteration_period_completed(self, engine, values):
        pass

    def on_before_epoch_completed(self, engine):
        return None

    def on_epoch_completed(self, engine, values):
        pass


class KeyTransformer(BaseTransformer):

    def __init__(self, output_keys):
        self.output_keys = output_keys

    def on_iteration_completed(self, engine, batch, endpoint):
        it_values = {}

        for key in self.output_keys:
            if key in endpoint:
                it_values[key] = endpoint[key].numpy()

            elif key in batch:
                it_values[key] = batch[key].numpy()

        return it_values

    def on_iteration_period_completed(self, engine, values):
        for key, value in values.items():
            engine.state.metrics[key] = np.mean(value)


class PairMetricTransformer(BaseTransformer):

    def __init__(self, entity_id, metric_config):
        self.entity_id = entity_id
        self.reduce = metric_config.get(REDUCE)
        self.output_keys = metric_config.get(OUTPUT_KEYS)

    def on_iteration_completed(self, engine, batch, endpoint):
        it_values = {}

        if not self.reduce:
            for e_id in self.entity_id:
                it_values[e_id] = batch[e_id]

        return it_values

    def on_epoch_completed(self, engine, values):
        if self.reduce:
            for metric_name in self.output_keys:
                reduced_values = self._reduce(engine, metric_name, values)

                for key, value in reduced_values.items():
                    engine.state.metrics[key] = value

        else:
            for e_id in self.entity_id:
                engine.state.metrics[e_id] = values[e_id]

            for metric_name in self.output_keys:
                for key, value in values.items():
                    if metric_name in key:
                        engine.state.metrics[key] = value

    def _reduce(self, engine, metric_name, values):
        reduced_values = {}

        for k, v in values.items():
            if metric_name in k:
                reduced_values[k] = np.mean(v)

        return reduced_values


class AsyncPairMetricTransformer(BaseTransformer):

    def __init__(self, entity_id, metric_config):
        self.entity_id = entity_id
        self.metric_config = metric_config

        self.reduce = metric_config.reduce
        self.output_keys = metric_config.output_keys

        self.num_processes = metric_config.num_processes

        self.pool = None
        self.async_results = None
        self.iteration_values = None

    def on_epoch_started(self, engine):
        self.pool = Pool(self.num_processes, maxtasksperchild=3)
        self.async_results = deque()
        self.iteration_values = []

    def on_iteration_completed(self, engine, batch, endpoint):
        return self._get_iteration_values()

    def on_before_epoch_completed(self, engine):
        while len(self.async_results) != 0:
            self.iteration_values.append(self.async_results.popleft().get())

        return self._get_iteration_values()

    def on_epoch_completed(self, engine, values):
        self.pool.close()
        self.pool.join()

        if self.reduce:
            for metric_name in self.output_keys:
                reduced_values = self._reduce(engine, metric_name, values)

                for key, value in reduced_values.items():
                    engine.state.metrics[key] = value

        else:
            for e_id in self.entity_id:
                engine.state.metrics[e_id] = values[e_id]

            for metric_name in self.output_keys:
                for key, value in values.items():
                    if metric_name in key:
                        engine.state.metrics[key] = value

    def _submit_request(self, r):
        self.async_results.append(self.pool.apply_async(submit_request_impl, (r,)))

        if len(self.async_results) >= self.num_processes:
            self.iteration_values.append(self.async_results.popleft().get())

            while len(self.async_results) != 0 and self.async_results[0].ready():
                self.iteration_values.append(self.async_results.popleft().get())

    def _get_iteration_values(self):
        if len(self.iteration_values) != 0:
            values = list2dict(self.iteration_values)
            self.iteration_values = []

            return values

        else:
            return None


"""
Transformer handler
"""


class TransformerHandler:

    def __init__(self, transformer, log_interval=None):
        self.transformer = transformer

        self.values = []
        self.iter_counter = 0

        self.log_interval = log_interval

    def epoch_started(self, engine):
        self.values = []
        self.iter_counter = 0

        self.transformer.on_epoch_started(engine)

    def iteration_started(self, engine):
        if self.iter_counter % self.log_interval == 0:
            self.values = []
            self.iter_counter = 0

    def iteration_completed(self, engine):
        it_value = self.transformer.on_iteration_completed(engine, engine.state.batch, engine.state.output)

        if it_value is not None:
            self.values.append(it_value)

        self.iter_counter += 1

    def on_iteration_period_completed(self, engine):
        self.transformer.on_iteration_period_completed(engine, list2dict(self.values))

    def on_epoch_completed(self, engine):
        it_value = self.transformer.on_before_epoch_completed(engine)

        if it_value is not None:
            self.values.append(it_value)

        self.transformer.on_epoch_completed(engine, list2dict(self.values))

    def attach(self, engine, name):
        """
        :param engine: Engine object
        :param name: list of names, unique identifier
        """

        if self.log_interval != -1:
            custom_state = f"{'_'.join(name)}_iteration"
            periodic_event_name = f"{custom_state.upper()}_FINISHED"

            def on_periodic_event(engine_p):
                if engine_p.state.iteration % self.log_interval == 0:
                    engine_p.fire_event(periodic_event_name)

            engine.register_events(periodic_event_name, event_to_attr={periodic_event_name: custom_state})
            engine.add_event_handler(Events.ITERATION_COMPLETED, on_periodic_event)

            engine.add_event_handler(Events.ITERATION_STARTED, self.iteration_started)
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
            engine.add_event_handler(periodic_event_name, self.on_iteration_period_completed)

        else:
            engine.add_event_handler(Events.EPOCH_STARTED, self.epoch_started)
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.on_epoch_completed)


"""
Async request
"""

class AsyncRequest:

    def __init__(self, entity_id, output_keys):
        self.entity_id = entity_id
        self.output_keys = output_keys

        self.state = {}
        self.state_keys = None

    def update_state(self, input_dict, start, end):
        if self.state_keys is None:
            self.state_keys = self._get_state_keys()

        for key in self.state_keys:
            if key in input_dict:
                v = input_dict[key]

                if key in self.state:
                    if isinstance(v, list):
                        self.state[key] = self.state[key] + v[start:end]

                    elif isinstance(v, np.ndarray):
                        self.state[key] = np.concatenate([self.state[key],
                                                          v[start:end]],
                                                         axis=0)

                    else:
                        self.state[key] = np.concatenate([self.state[key],
                                                          v[start:end].cpu().numpy()],
                                                         axis=0)

                else:
                    if isinstance(v, list) or isinstance(v, np.ndarray):
                        self.state[key] = v[start:end]

                    else:
                        self.state[key] = v[start:end].cpu().numpy()

    def __call__(self):
        values = self._process_request()

        if self.entity_id is not None:
            for key in self.entity_id:
                values[key] = self.state[key]

        return values

    def _get_state_keys(self):
        return set(self.entity_id) if self.entity_id is not None else set()

    @abstractmethod
    def _process_request(self):
        ...


"""
Support utils
"""

def submit_request_impl(r):
    return r()

def list2dict(l):
    d = {}

    if len(l) == 0:
        return d

    for k in l[0].keys():
        d[k] = []

        for v in l:
            d[k].append(v[k])

        if isinstance(d[k][0], list):
            d[k] = [j for i in d[k] for j in i]

        elif isinstance(d[k][0], np.ndarray):
            if d[k][0].ndim == 0:
                d[k] = np.stack(d[k])

            else:
                d[k] = np.concatenate(d[k])

        elif isinstance(d[k][0], dict) and len(l) == 1:
            d[k] = d[k][0]

        elif isinstance(d[k][0], float):
            pass

        else:
            raise NotImplementedError()

    return d


"""
Legacy code
"""

#     for e_id in self.entity_id:
#         engine.state.metrics[e_id] = values[e_id]
#
#     for metric_name in self.output_keys:
#         for key, value in values.items():
#             if metric_name in key:
#                 engine.state.metrics[key] = value


# Create periodic event
# custom_state = f"{'_'.join(name)}_iteration"
# periodic_event_name = f"{custom_state.upper()}_FINISHED"
# setattr(self, "Events", Enum("Events", periodic_event_name))
#
# for e in self.Events:
#     State.event_to_attr[e] = custom_state
#
# periodic_event = getattr(self.Events, periodic_event_name)
#
# def on_periodic_event(engine_p):
#     if engine_p.state.iteration % self.log_interval == 0:
#         engine_p.fire_event(periodic_event)
#
# engine.register_events(*self.Events)
# engine.add_event_handler(Events.ITERATION_COMPLETED, on_periodic_event)
#
# engine.add_event_handler(periodic_event, self.completed)

