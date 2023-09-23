from source.evaluation.classical.namespace import *
from source.evaluation.rel_pose.namespace import *
from source.evaluation.bag_rel_pose.namespace import *
from source.evaluation.visual_localization.namespace import *

EVAL_TAG = 'eval_tag'
DEFAULT_EVAL_TAG = ''
ICCV2023_EVAL_TAG = '#ICCV2023'
REFERENCE_EVAL_TAG = '#REFERENCE'
REFERENCE1_EVAL_TAG = '#REFERENCE1'
CHECK_EVAL_TAG = '#CHECK'

"""
Evaluation tasks
"""

FEATURES = 'features'
REL_POSE = 'rel_pose'
BAG_REL_POSE = 'bag_rel_pose'
VISUAL_LOCALIZATION = 'visual_localization'
CLASSICAL = 'classical'
TIME = 'time'

"""
Transformers variables
"""

R_ERR = 'r_err'
T_ERR = 't_err'
HCR_ERR = 'hcr_err'

R_ERR_THRESH = 'r_err_thresh'
T_ERR_THRESH = 't_err_thresh'
HCR_ERR_THRESH = 'hcr_err_thresh'

R_mAA = 'r_mAA'
T_mAA = 't_mAA'
HCR_mAA = 'hcr_mAA'

LOWE_RATIO = 'lowe_ratio'
INL_THRESH = 'inl_thresh'
ESTIMATOR = 'estimator'
CONFIDENCE = 'confidence'
NUM_RANSAC_ITER = 'num_ransac_iter'
MIN_NUM_INLIERS = 'min_num_inliers'

HTUNE = 'htune'
HTUNE_PREFIX = 'htune_'
HTUNE_LOWE_RATIO = f'{HTUNE_PREFIX}{LOWE_RATIO}'
HTUNE_INL_THRESH = f'{HTUNE_PREFIX}{INL_THRESH}'

TEST = 'test'

ID = 'id'

INVALID_KP = 0.0
MAX_ANGLE_ERR = 180.0

"""
Visualization variables
"""

EVALUATION_TASK = 'evaluation_task'
BACKEND = 'backend'
METHODS = 'methods'

IMPORT_EVALUATION_TASK = 'import_evaluation_task'
IMPORT_BACKEND = 'import_backend'
IMPORT_EVAL_TAG = 'import_eval_tag'

PLOT_PARAMS = 'plot_params'

HTUNE2NAME = {LOWE_RATIO: 'Lowe ratio',
              INL_THRESH: 'Inlier threshold'}

COLOR = 'color'
LINE_STYLE = 'line_style'
ALIAS = 'alias'
REL_PATH = 'rel_path'
SAVE_NAME = 'save_name'
FONT_SIZE = 'font_size'
LEGEND_LOC = 'legend_loc'

PLOT_TITLE = 'plot_title'
TITLE_FONT_SIZE = 'title_font_size'
BEST = 'best'
SAVE = 'save'
SAVE_DIR = 'save_dir'

# HCR_ACC = 'hcr_acc'
