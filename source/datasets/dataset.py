from omegaconf import OmegaConf
from hydra import compose, initialize

from torch.utils.data.dataloader import DataLoader
from torch.utils.data._utils.collate import default_collate

import source.datasets.base.utils as du
import source.datasets.megadepth.utils as mdu
import source.evaluation.namespace as eva_ns

from source.models.model import get_input_size_divisor

from source.datasets.base.transforms import ImageDepthTFactory, ImageDepthCalibFeaturesTFactory, ImageHCalibTFactory
from source.datasets.megadepth.transforms import get_megadepth_transforms

from source.datasets.megadepth.dataset import MegaDepthDataset
from source.datasets.imcpt.dataset import IMCPTDataset
from source.datasets.scannet.dataset import ScanNetDataset
from source.datasets.hpatches.dataset import HPatchesDataset
from source.datasets.aachen.dataset import AachenDataset


def create_dataset(dataset_name, dataset_mode_config, models_configs, loops, config):
    input_size_divisor = get_input_size_divisor(models_configs)
    backend = config.datasets.get(eva_ns.BACKEND)

    if dataset_name == du.MEGADEPTH:
        item_transforms = get_megadepth_transforms(dataset_mode_config, input_size_divisor)

        return MegaDepthDataset.from_config(dataset_mode_config, backend, item_transforms)

    elif dataset_name == du.IMC_PT:
        item_transforms = ImageDepthCalibFeaturesTFactory(dataset_mode_config.transforms, input_size_divisor).create()

        return IMCPTDataset.from_config(dataset_mode_config, backend, item_transforms)

    elif dataset_name == du.SCANNET:
        item_transforms = ImageDepthCalibFeaturesTFactory(dataset_mode_config.transforms, input_size_divisor).create()

        return ScanNetDataset.from_config(dataset_mode_config, backend, item_transforms)

    elif dataset_name == du.HPATCHES:
        item_transforms = ImageHCalibTFactory(dataset_mode_config.transforms, input_size_divisor).create()

        return HPatchesDataset.from_config(dataset_mode_config, backend, item_transforms)

    elif dataset_name == du.AACHEN:
        item_transforms = ImageDepthCalibFeaturesTFactory(dataset_mode_config.transforms, input_size_divisor).create()

        return AachenDataset.from_config(dataset_mode_config, backend, item_transforms)

    else:
        raise ValueError(f"No such dataset: {dataset_name}")


def create_loader(dataset, loader_mode_config):
    batch_size = loader_mode_config.batch_size
    num_workers = loader_mode_config.get(du.NUM_WORKERS, 0)
    num_samples = loader_mode_config.get(du.NUM_SAMPLES, -1)
    sampler_type = loader_mode_config.get(du.SAMPLER)
    collate = loader_mode_config.get(du.COLLATE)

    if collate is None:
        collate_fn = default_collate

    else:
        raise NotImplementedError

    if sampler_type == du.SUBSET_SAMPLER:
        shuffle = loader_mode_config.get(du.SHUFFLE, False)

        sampler = du.SubsetSampler(dataset, num_samples, shuffle)

    elif sampler_type == du.START_SEQ_SAMPLER:
        start_from = loader_mode_config.get(du.START_FROM, 0)

        sampler = du.StartSeqSampler(dataset, num_samples, start_from)

    elif sampler_type == mdu.SCENE_SAMPLER:
        sampler = mdu.SceneSampler(dataset)

    elif sampler_type == mdu.CORRS_SAMPLER:
        sampler = mdu.CorrsSampler(dataset, loader_mode_config['pair_csv_path'])

    elif sampler_type is None and num_samples == -1:
        sampler = None

    else:
        raise NotImplementedError

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=collate_fn)

    return data_loader


"""
Legacy code
"""

# elif dataset_name == du.SAMSUNG_OFFICE:
#     item_transforms = SamsungOfficeTransformsFactory(dataset_mode_config.transforms, input_size_divisor).create()
#
#     return SamsungOfficeDataset.from_config(dataset_mode_config, item_transforms)

# elif dataset_name == du.MROB_LAB:
#     item_transforms = ImageDepthTFactory(dataset_mode_config.transforms, input_size_divisor).create()
#
#     return MRobLabDataset.from_config(dataset_mode_config, item_transforms)

# in_memory = dataset_mode_config.get(d.IN_MEMORY)
#
# if in_memory is not None and in_memory:
#     return InMemoryDataset.from_config(loops, dataset_mode_config)
#
# else:

# from source.core.dataset import InMemoryDataset

# from source.datasets.samsung_office.transforms import SamsungOfficeTransformsFactory

# from source.datasets.samsung_office.dataset import SamsungOfficeDataset
# from source.datasets.mrob_lab.dataset import MRobLabDataset

# def create_dummy_config(dataset_name, mode,
#                         config_name=None,
#                         loader_config=None, config_overrides=None):
#     with initialize(config_path="../config"):
#         config = compose(overrides=[f"+datasets/{dataset_name}={mode if config_name is None else config_name}"])
#
#         OmegaConf.set_struct(config, False)
#
#         if loader_config is None:
#             mode_loader_config = {f'{mode}': {'batch_size': 1,
#                                               'num_samples': 1,
#                                               'sampler': 'start_seq',
#                                               'start_from': 0,
#                                               'num_workers': 0}
#                                   }
#
#         else:
#             mode_loader_config = {f'{mode}': loader_config}
#
#         config['datasets']['dataset_name'] = dataset_name
#         config['datasets'][dataset_name]['loader'] = mode_loader_config
#
#         if config_overrides is not None:
#             config = OmegaConf.merge(config, config_overrides)
#
#         print(OmegaConf.to_yaml(config))
#
#     return config
#
#
# def create_dummy_iterator(config, cfg_mode):
#     dataset_name = config.datasets.dataset_name
#     dataset_config = config.datasets[dataset_name]
#
#     dataset_mode_config = dataset_config[cfg_mode]
#     loader_mode_config = dataset_config.loader[cfg_mode]
#
#     dataset = create_dataset(dataset_name, dataset_mode_config, {}, None)
#     loader = create_loader(dataset, loader_mode_config)
#
#     return loader.__iter__()

# from source.datasets.hpatches.hpatches_transforms import get_hpatches_test_transforms
# from source.datasets.aachen.aachen_transforms import get_aachen_test_transforms
# from source.datasets.imb.imb_transforms import get_imb_test_transform
#
# from source.datasets.hpatches.hpatches_dataset import HPatchesDataset
# from source.datasets.aachen.aachen_dataset import AachenDataset
# from source.datasets.imb.imb_dataset import IMBDataset

# if dataset_name == du.HPATCHES:
#     item_transforms = get_hpatches_test_transforms(dataset_mode_config, input_channels, input_size_divisor)
#
#     return HPatchesDataset.from_config(dataset_mode_config, item_transforms)
# elif dataset_name == du.AACHEN:
#     item_transforms = get_aachen_test_transforms(dataset_mode_config, input_channels, input_size_divisor)
#
#     return AachenDataset.from_config(dataset_mode_config, item_transforms)
#
# elif dataset_name == du.IMB:
#     item_transforms = get_imb_test_transform(dataset_mode_config, input_channels, input_size_divisor)
#
#     return IMBDataset.from_config(dataset_mode_config, item_transforms)

# if sampler == du.SCENE_SAMPLER:
#     sampler = mdu.MegaDepthSceneSampler(dataset, batch_size, num_samples, shuffle, parent_engine)
# if collate == du.MEMORY_BANK_COLLATE:
#     collate_fn = mdu.memory_bank_collate
#         elif mode in [lp.RUN]:
#             # item_transforms = get_megadepth_train_transforms(d_config, input_channels)
#             item_transforms = get_megadepth_test_transforms(d_config, input_channels)
#             # item_transforms = get_megadepth_labeling_transforms(d_config, input_channels)
#
#         else:
#             raise NotImplementedError

    #
    #     elif d_key == du.MEGADEPTH_HA:
    #         if mode in [lp.RUN]:
    #             item_transforms = get_megadepth_ha_transforms(d_config)
    #
    #         else:
    #             raise NotImplementedError
    #
    #         dataset = MegaDepthHADataset.from_config(d_config, item_transforms)