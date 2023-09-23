# Add the commands below to the first cell of the notebook to have all necessary imports

# %run __init__.py
# %load_ext autoreload
# %autoreload 2

import os
import sys

module_path = "/home/konstantin/personal/Summertime/"
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

import source.datasets.base.utils as du
import source.utils.endpoint_utils as eu
import source.evaluation.namespace as eva_ns

from source.experiment import SummertimeExperiment
from source.evaluation.classical.vis import draw_cv_keypoints, draw_cv_matches
from source.datasets.base.utils import RBTDataWrapper, HDataWrapper
from source.utils.endpoint_utils import grid2flat, flat2grid
from source.utils.vis_utils import plot_figures, torch2cv


