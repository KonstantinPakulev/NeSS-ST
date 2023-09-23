import os
import shutil
import numpy as np

import source.datasets.base.utils as du
import source.evaluation.namespace as ns
import source.utils.endpoint_utils as eu
import source.pose.estimators.colmap.utils as cu
import source.datasets.aachen.preprocessing.annotations as aa_an

from source.core.evaluation import BaseTransformer
from source.pose.matchers.factory import instantiate_matcher
from source.pose.estimators.colmap.estimator import COLMAPEstimator
from source.pose.estimators.colmap.utils import Camera, Image
from source.utils.common_utils import qvec2rotmat
from source.evaluation.utils import get_kp_desc_and_kp_desc_mask
from source.datasets.aachen.preprocessing.annotations import get_intrinsics_path, get_extrinsics_path, \
    get_matches_list_file_path, get_cameras_iterator, get_images_iterator


REFERENCE_MODEL_DIR = 'reference_model'

IMAGES_REL_PATH = 'images/images_upright/'

TRIANGULATED_MODEL_DIR = 'triangulated_model'
REGISTERED_MODEL = 'registered_model'

QUERIES_REL_PATH = 'queries/night_time_queries_with_intrinsics.txt'


class AachenVisualLocalizationTransformer(BaseTransformer):

    def __init__(self, dataset_mode_config, model_mode_eval_params, device):
        self.dataset_mode_config = dataset_mode_config
        self.model_mode_eval_params = model_mode_eval_params
        self.device = device

        self.version = self.dataset_mode_config.version
        self.dataset_path = self.dataset_mode_config.dataset_path
        self.rec_id = ns.AACHEN_REC_ID.format(self.version)

        self.matcher = instantiate_matcher(self.model_mode_eval_params, device)

        self.estimator = COLMAPEstimator.from_config(self.rec_id, self.model_mode_eval_params)

        reference_db_path = os.path.join(self.dataset_path, get_db_file(self.version))
        name2im_cam_id = self.estimator.create_from_reference_database(reference_db_path)

        cameras = read_cameras(get_intrinsics_path(self.dataset_path, self.version), name2im_cam_id)
        images = read_images(get_extrinsics_path(self.dataset_path, self.version), name2im_cam_id)

        self.reference_model_path = os.path.join(self.estimator.rec_path, REFERENCE_MODEL_DIR)

        create_reference_model(self.reference_model_path, cameras, images)

    def on_iteration_completed(self, engine, batch, endpoint):
        image_name1, image_name2 = batch[du.IMAGE_NAME1], batch[du.IMAGE_NAME2]

        kp1, kp2 = batch[eu.KP1], batch[eu.KP2]
        kp_desc1, kp_desc2, kp_desc_mask1, kp_desc_mask2 = get_kp_desc_and_kp_desc_mask(batch)

        mm_desc_mask1, nn_desc_idx1 = self.matcher.match(kp_desc1, kp_desc2,
                                                         kp_desc_mask1, kp_desc_mask2)

        image_name1 = [i_n.split(IMAGES_REL_PATH)[1] for i_n in batch[du.IMAGE_NAME1]]
        image_name2 = [i_n.split(IMAGES_REL_PATH)[1] for i_n in batch[du.IMAGE_NAME2]]

        self.estimator.import_keypoints_and_matches(image_name1, image_name2,
                                                    batch[eu.KP1].numpy(), batch[eu.KP2].numpy(),
                                                    mm_desc_mask1.cpu().numpy(), nn_desc_idx1.cpu().numpy())

    def on_before_epoch_completed(self, engine):
        self.estimator.run_matches_importer(get_matches_list_file_path(self.dataset_path, self.version))

        triangulated_model_path = get_triangulated_model_path(self.estimator.rec_path)

        self.estimator.run_point_triangulator(os.path.join(self.dataset_path, IMAGES_REL_PATH),
                                              self.reference_model_path, triangulated_model_path)

        registered_model_path = get_registered_model_path(self.estimator.rec_path)

        self.estimator.run_image_registrator(triangulated_model_path, registered_model_path)
        self.estimator.run_model_converter(registered_model_path)

        query_names = get_query_names(self.dataset_path)
        query_extrinsics = get_query_extrinsics(query_names, registered_model_path)

        return {self.rec_id: query_extrinsics}

    def on_epoch_completed(self, engine, values):
        engine.state.metrics[self.rec_id] = values[self.rec_id]


"""
Support utils
"""

def get_db_file(version):
    if version == aa_an.V1_0:
        return 'database_v1_0.db'

    elif version == aa_an.V1_1:
        return 'database_v1_1.db'

    else:
        raise ValueError(version)


def get_query_extrinsics(query_names, model_path):
    query_extrinsics = {}

    with open(os.path.join(model_path, cu.IMAGES_TXT_FILE)) as f:
        lines = f.readlines()

        for line in lines[4::2]:
            line = line.strip('\n').split(' ')

            image_name = line[-1]

            if image_name in query_names:
                query_extrinsics[image_name.split('/')[-1]] = ' '.join(line[1:-2])

    return query_extrinsics


def get_query_names(dataset_path):
    query_image_list_path = os.path.join(dataset_path, QUERIES_REL_PATH)

    query_names = set()

    with open(query_image_list_path) as f:
        raw_queries = f.readlines()

        for q in raw_queries:
            q = q.strip('\n').split(' ')
            query_names.add(q[0])

    return query_names


def get_triangulated_model_path(rec_dir_path):
    triangulated_model_path = os.path.join(rec_dir_path, TRIANGULATED_MODEL_DIR)

    if os.path.exists(triangulated_model_path):
        shutil.rmtree(triangulated_model_path)

    os.mkdir(triangulated_model_path)

    return triangulated_model_path


def get_registered_model_path(rec_dir_path):
    registered_model_path = os.path.join(rec_dir_path, REGISTERED_MODEL)

    if os.path.exists(registered_model_path):
        shutil.rmtree(registered_model_path)

    os.mkdir(registered_model_path)

    return registered_model_path


def create_reference_model(reference_model_path, cameras, images):
    if os.path.exists(reference_model_path):
        shutil.rmtree(reference_model_path)

    os.mkdir(reference_model_path)

    with open(os.path.join(reference_model_path, cu.CAMERAS_TXT_FILE), 'w') as f:
        for camera in cameras.values():
            f.write('%d %s %s %s %s\n' % (
                camera.id,
                camera.model.model_name,
                camera.width,
                camera.height,
                ' '.join(map(str, camera.params))
            ))

    with open(os.path.join(reference_model_path, cu.IMAGES_TXT_FILE), 'w') as f:
        for image in images.values():
            f.write('%d %s %s %d %s\n\n' % (
                image.id,
                ' '.join(map(str, image.qvec)),
                ' '.join(map(str, image.tvec)),
                image.camera_id,
                image.name
            ))

    with open(os.path.join(reference_model_path, cu.POINTS_TXT_FILE), 'w') as f:
        pass


def read_cameras(intrinsics_path, name2im_cam_id):
    cameras = {}

    cameras_iter = get_cameras_iterator(intrinsics_path)

    for i in cameras_iter:
        image_name, camera_model_name, width, height, params = i

        cameras[image_name] = Camera(id=name2im_cam_id[image_name][1],
                                     model=cu.CAMERA_MODEL_NAMES[camera_model_name],
                                     width=width,
                                     height=height,
                                     params=params)

    cameras = dict(sorted(cameras.items(), key=lambda x: x[1].id))

    return cameras


def read_images(extrinsics_path, name2im_cam_id):
    images = {}

    images_iter = get_images_iterator(extrinsics_path)

    for i in images_iter:
        image_name, qvec, tvec = i

        images[image_name] = Image(id=name2im_cam_id[image_name][0],
                                   qvec=qvec, tvec=tvec,
                                   camera_id=name2im_cam_id[image_name][1], name=image_name,
                                   xys=None, point3D_ids=None)

    images = dict(sorted(images.items(), key=lambda x: x[1].id))

    return images
