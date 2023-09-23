import os
import numpy as np

from PIL import Image

from source.datasets.base import utils as du
from source.utils import endpoint_utils as eu
import source.evaluation.namespace as eva_ns
import source.evaluation.bag_rel_pose.namespace as brp_ns

from source.core.evaluation import AsyncRequest
from source.datasets.imcpt.preprocessing.annotations import get_scene_path, get_scene_bags_path, \
    get_scene_bag_pairs_path
from source.evaluation.bag_rel_pose.metrics import relative_pose_error

MAPPED_MODEL_DIR = 'mapped_model'


class BagData:

    def __init__(self, dataset_path, scene_name, bag_id):
        self.bag_id = bag_id

        scene_path = get_scene_path(dataset_path, scene_name)
        scene_bags_path = get_scene_bags_path(scene_path)

        with open(os.path.join(scene_bags_path, f'{bag_id}.txt')) as file:
            bag_image_paths = [line.rstrip() for line in file]

        self.image_names = [os.path.basename(bip) for bip in bag_image_paths]
        self.image_sizes = [Image.open(os.path.join(scene_path, bip)).size for bip in bag_image_paths]

        self.images_path = os.path.join(scene_path, 'images')
        self.bag_image_pairs_path = os.path.join(get_scene_bag_pairs_path(scene_path), f"{bag_id}.txt")

    def fits_in_batch(self):
        return len(self.image_names) <= 256


class FRequest(AsyncRequest):

    def __init__(self, estimator):
        super().__init__(None, None)
        self.estimator = estimator

    def update(self, batch, bundle, i):
        super().update_state(batch, i, i + 1)
        super().update_state(bundle, i, i + 1)

    def _get_state_keys(self):
        state_keys = super()._get_state_keys()
        state_keys.update([eu.KP1,
                           du.INTRINSICS1, du.INTRINSICS2,
                           brp_ns.NN_KP2, brp_ns.MM_DESC_MASK1])

        return state_keys

    def _process_request(self):
        F, inl_mask = self.estimator.estimate(self.state[eu.KP1][0], self.state[brp_ns.NN_KP2][0],
                                              self.state[brp_ns.MM_DESC_MASK1][0],
                                              self.state[du.INTRINSICS1][0], self.state[du.INTRINSICS2][0],
                                              recover_pose=False)

        values = {brp_ns.F_ESTIMATE: F,
                  brp_ns.INL_MASK: inl_mask}

        return values


class BagRelPoseRequest(AsyncRequest):

    def __init__(self, entity_id, output_keys, estimator, bag_data):
        super().__init__(entity_id, output_keys)
        self.estimator = estimator
        self.bag_data = bag_data

        if not self.bag_data.fits_in_batch():
            self.estimator.create_new_database()
            self.estimator.initialize_images_and_cameras(self.bag_data.image_names, self.bag_data.image_sizes)

    def update(self, batch, bundle, start, end):
        if self.bag_data.fits_in_batch():
            super().update_state(batch, start, end)
            super().update_state(bundle, start, end)

        else:
            super().update_state(batch, start, end)

            image_name1 = get_basenames(batch[du.IMAGE_NAME1])[start:end]
            image_name2 = get_basenames(batch[du.IMAGE_NAME2])[start:end]

            nn_desc_idx1 = bundle[brp_ns.NN_DESC_IDX1][start:end].cpu().numpy()

            self.estimator.import_keypoints_and_matches(image_name1, image_name2,
                                                        batch[eu.KP1][start:end].numpy(),
                                                        batch[eu.KP2][start:end].numpy(),
                                                        bundle[brp_ns.MM_DESC_MASK1][start:end].cpu().numpy(),
                                                        nn_desc_idx1)

            if eva_ns.ESTIMATOR in self.estimator.two_view_eval_params:
                self.estimator.import_two_view_geometries(image_name1, image_name2,
                                                          bundle[brp_ns.F_ESTIMATE][start:end],
                                                          bundle[brp_ns.INL_MASK][start:end],
                                                          nn_desc_idx1)

    def _get_state_keys(self):
        state_keys = super()._get_state_keys()

        if self.bag_data.fits_in_batch():
            state_keys.update([du.IMAGE_NAME1, du.IMAGE_NAME2,
                               eu.KP1, eu.KP2,
                               brp_ns.MM_DESC_MASK1, brp_ns.NN_DESC_IDX1,
                               brp_ns.F_ESTIMATE, brp_ns.INL_MASK,
                               du.EXTRINSICS1, du.EXTRINSICS2])

        else:
            state_keys.update([du.IMAGE_NAME1, du.IMAGE_NAME2,
                               du.EXTRINSICS1, du.EXTRINSICS2])

        return state_keys

    def _process_request(self):
        image_name1, image_name2 = get_basenames(self.state[du.IMAGE_NAME1]), get_basenames(self.state[du.IMAGE_NAME2])

        if self.bag_data.fits_in_batch():
            self.estimator.create_new_database()
            self.estimator.initialize_images_and_cameras(self.bag_data.image_names, self.bag_data.image_sizes)

            self.estimator.import_keypoints_and_matches(image_name1, image_name2,
                                                        self.state[eu.KP1], self.state[eu.KP2],
                                                        self.state[brp_ns.MM_DESC_MASK1],
                                                        self.state[brp_ns.NN_DESC_IDX1])

            if eva_ns.ESTIMATOR in self.estimator.two_view_eval_params:
                self.estimator.import_two_view_geometries(image_name1, image_name2,
                                                          self.state[brp_ns.F_ESTIMATE],
                                                          self.state[brp_ns.INL_MASK],
                                                          self.state[brp_ns.NN_DESC_IDX1])

        if eva_ns.ESTIMATOR not in self.estimator.two_view_eval_params:
            self.estimator.run_matches_importer(self.bag_data.bag_image_pairs_path)

        mapped_model_path = get_mapped_model_path(self.estimator.rec_path)

        self.estimator.run_mapper(self.bag_data.images_path, mapped_model_path)

        r_err, t_err = relative_pose_error(mapped_model_path,
                                           image_name1, image_name2,
                                           self.state[du.EXTRINSICS1], self.state[du.EXTRINSICS2],
                                           self.estimator.image_name2id)

        self.estimator.delete_reconstruction()

        values = {eva_ns.R_ERR: r_err,
                  eva_ns.T_ERR: t_err}

        return values


"""
Support utils
"""


def get_mapped_model_path(rec_dir_path):
    mapped_model_path = os.path.join(rec_dir_path, MAPPED_MODEL_DIR)

    if os.path.exists(mapped_model_path):
        shutil.rmtree(mapped_model_path)

    os.mkdir(mapped_model_path)

    return mapped_model_path


def get_basenames(l):
    return [os.path.basename(i) for i in l]
