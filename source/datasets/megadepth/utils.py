import pandas as pd

from torch.utils.data.sampler import Sampler

import source.datasets.base.utils as du

from source.datasets.base.utils import from_pairs_annotations

COUNTS = 'counts'

SCENE_SAMPLER = 'scene'
CORRS_SAMPLER = 'corrs'


class SceneSampler(Sampler):

    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

        annotations = data_source.annotations
        scene_name = annotations.groupby(du.SCENE_NAME)[du.SCENE_NAME].count().reset_index(name=COUNTS).\
            sample(1, weights=COUNTS)[du.SCENE_NAME].item()

        self.indices = annotations[annotations[du.SCENE_NAME] == scene_name].index.to_numpy()

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class CorrsSampler(Sampler):

    def __init__(self, data_source, pair_csv_path):
        super().__init__(data_source)
        self.data_source = data_source

        annotations = data_source.annotations
        s_ann = annotations.sample(1)

        scene_name = s_ann[du.SCENE_NAME].item()
        image1 = s_ann[du.IMAGE1].item()

        pair_csv = pd.read_csv(pair_csv_path, index_col=[0])
        pairs = pair_csv.loc[((pair_csv[du.SCENE_NAME] == scene_name) & (pair_csv[du.IMAGE1] == image1)) |
                             ((pair_csv[du.SCENE_NAME] == scene_name) & (pair_csv[du.IMAGE2] == image1))]
        corrs = from_pairs_annotations(pairs)

        self.indices = annotations[annotations[du.IMAGE1].isin(corrs[du.IMAGE1])].index

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
