import os
import shutil
import subprocess
import numpy as np

import source.pose.estimators.colmap.utils as cu
import source.pose.estimators.namespace as est_ns
import source.evaluation.namespace as eva_ns

from source.pose.estimators.colmap.database import COLMAPDatabase
from source.pose.estimators.colmap.utils import image_ids_to_pair_id, pair_id_to_image_ids


DB_FILE = 'database.db'

ABS_POSE = 'abs_pose'


class COLMAPEstimator:

    @staticmethod
    def from_config(rec_id, model_mode_eval_params):
        two_view_eval_params = model_mode_eval_params.estimator.two_view
        abs_pose_eval_params = model_mode_eval_params.estimator.get(ABS_POSE)

        return COLMAPEstimator(rec_id,
                               model_mode_eval_params.estimator.exec_path,
                               two_view_eval_params,
                               abs_pose_eval_params)

    def __init__(self, rec_id,
                 exec_path,
                 two_view_eval_params,
                 abs_pose_eval_params):
        self.exec_path = exec_path
        self.rec_path = os.path.join(os.getcwd(), rec_id)

        if os.path.exists(self.rec_path):
            shutil.rmtree(self.rec_path)

        os.mkdir(self.rec_path)

        self.rec_db_path = os.path.join(self.rec_path, DB_FILE)

        self.image_name2id = {}
        self.kp_image_ids = set()
        self.matches_pair_ids = set()
        self.two_view_pair_ids = set()

        self.two_view_eval_params = two_view_eval_params
        self.abs_pose_eval_params = abs_pose_eval_params

    def create_new_database(self):
        db = COLMAPDatabase.connect(self.rec_db_path)
        db.create_tables()
        db.close()

    def create_from_reference_database(self, reference_db_path):
        shutil.copyfile(reference_db_path, self.rec_db_path)

        db = COLMAPDatabase.connect(self.rec_db_path)

        name2im_cam_id = {}

        rows = db.get_image_and_camera_ids()

        for r in rows:
            name2im_cam_id[r[0]] = (r[1], r[2])

        db.close()

        for k, v in name2im_cam_id.items():
            self.image_name2id[k] = v[0]

        return name2im_cam_id

    def initialize_images_and_cameras(self, image_names, image_sizes):
        db = COLMAPDatabase.connect(self.rec_db_path)

        for i, (im_n, im_s) in enumerate(zip(image_names, image_sizes)):
            image_id = i + 1

            w, h = im_s[0], im_s[1]
            params = [max(w, h) * cu.DEFAULT_FOCAL_LENGTH_FACTOR, w / 2, h / 2, 0]

            db.insert_camera(cu.SIMPLE_RADIAL_ID, w, h, params)
            db.insert_image(im_n, image_id)

            self.image_name2id[im_n] = image_id

        db.commit()
        db.close()

    def import_keypoints_and_matches(self, image_name1, image_name2,
                                     kp1, kp2,
                                     mm_desc_mask1, nn_desc_idx1):
        db = COLMAPDatabase.connect(self.rec_db_path)

        for image_name1i, image_name2i,\
            kp1i, kp2i, \
            mm_desc_mask1i, nn_desc_idx1i in zip(image_name1, image_name2,
                                                  kp1, kp2,
                                                  mm_desc_mask1, nn_desc_idx1):
            image_id1 = self.image_name2id[image_name1i]
            image_id2 = self.image_name2id[image_name2i]

            for image_idj, kpj in zip([image_id1, image_id2], [kp1i, kp2i]):
                if image_idj not in self.kp_image_ids:
                    db.insert_keypoints(image_idj, kpj)
                    self.kp_image_ids.add(image_idj)

            pair_id = image_ids_to_pair_id(image_id1, image_id2)

            if pair_id in self.matches_pair_ids:
                continue

            matches = np.stack([np.arange(0, nn_desc_idx1i.shape[0]),
                                nn_desc_idx1i], axis=-1)
            matches = matches[mm_desc_mask1i]

            if image_id1 > image_id2:
                matches = matches[:, [1, 0]]

            db.insert_matches(pair_id, matches)

            self.matches_pair_ids.add(pair_id)

        db.commit()
        db.close()

    def import_two_view_geometries(self, image_name1, image_name2, F, inl_mask, nn_desc_idx1):
        db = COLMAPDatabase.connect(self.rec_db_path)

        for image_name1i, image_name2i,\
            Fi, inl_maski,\
            nn_desc_idx1i in zip(image_name1, image_name2,
                                 F, inl_mask,
                                 nn_desc_idx1):
            image_id1 = self.image_name2id[image_name1i]
            image_id2 = self.image_name2id[image_name2i]

            pair_id = image_ids_to_pair_id(image_id1, image_id2)

            if pair_id in self.two_view_pair_ids:
                continue

            matches = np.stack([np.arange(0, nn_desc_idx1i.shape[0]),
                                nn_desc_idx1i], axis=-1)
            matches = matches[inl_maski]

            if image_id1 > image_id2:
                matches = matches[:, [1, 0]]

            db.insert_two_view_geometry(pair_id, matches, F=Fi)

            self.two_view_pair_ids.add(pair_id)

        db.commit()
        db.close()

    def run_matches_importer(self, matches_list_file_path):
        command = [self.exec_path, 'matches_importer']
        command += ['--database_path', self.rec_db_path]
        command += ['--match_list_path', matches_list_file_path]
        command += ['--match_type', 'pairs']
        command += ['--SiftMatching.use_gpu', '0']

        if self.two_view_eval_params is not None:
            if eva_ns.INL_THRESH in self.two_view_eval_params:
                command += ['--SiftMatching.max_error',
                            str(self.two_view_eval_params.inl_thresh)]

            if eva_ns.CONFIDENCE in self.two_view_eval_params:
                command += ['--SiftMatching.confidence',
                            str(self.two_view_eval_params.confidence)]

            if eva_ns.NUM_RANSAC_ITER in self.two_view_eval_params:
                command += ['--SiftMatching.max_num_trials',
                            str(self.two_view_eval_params.num_ransac_iter)]

            if eva_ns.MIN_NUM_INLIERS in self.two_view_eval_params:
                command += ['--SiftMatching.min_num_inliers',
                            str(self.two_view_eval_params.min_num_inliers)]

        subprocess.run(command)

    def run_mapper(self, images_path, output_path):
        command = [self.exec_path, 'mapper']
        command += ['--image_path', images_path]
        command += ['--database_path', self.rec_db_path]
        command += ['--output_path', output_path]
        command += ['--Mapper.min_model_size', '3']

        if self.abs_pose_eval_params is not None:
            if eva_ns.INL_THRESH in self.abs_pose_eval_params:
                command += ['--Mapper.abs_pose_max_error',
                            str(self.abs_pose_eval_params.inl_thresh)]

            if eva_ns.MIN_NUM_INLIERS in self.abs_pose_eval_params:
                command += ['--Mapper.abs_pose_min_num_inliers',
                            str(self.abs_pose_eval_params.min_num_inliers)]

        subprocess.run(command)

    def run_point_triangulator(self, images_path, input_path, output_path):
        command = [self.exec_path, 'point_triangulator']
        command += ['--image_path', images_path]
        command += ['--database_path', self.rec_db_path]
        command += ['--input_path', input_path]
        command += ['--output_path', output_path]
        command += ['--Mapper.ba_refine_focal_length', '0']
        command += ['--Mapper.ba_refine_principal_point', '0']
        command += ['--Mapper.ba_refine_extra_params', '0']

        subprocess.run(command)

    def run_image_registrator(self, input_path, output_path):
        command = [self.exec_path, 'image_registrator']
        command += ['--database_path', self.rec_db_path]
        command += ['--input_path', input_path]
        command += ['--output_path', output_path]
        command += ['--Mapper.ba_refine_focal_length', '0']
        command += ['--Mapper.ba_refine_principal_point', '0']
        command += ['--Mapper.ba_refine_extra_params', '0']

        subprocess.run(command)

    def run_model_converter(self, path):
        command = [self.exec_path, 'model_converter']
        command += ['--input_path', path]
        command += ['--output_path', path]
        command += ['--output_type', 'TXT']

        subprocess.run(command)

    def get_fund_mat_and_inliers_mask(self, image_name1, image_name2, num_kp):
        db = COLMAPDatabase.connect(self.rec_db_path)

        image_ids2row = {}

        rows = db.get_fund_mat_and_inliers()

        for r in rows:
            img_id1, img_id2 = pair_id_to_image_ids(r[0])

            inl_maski = np.zeros((num_kp), dtype=np.bool)

            if r[4] is not None:
                Fi = np.fromstring(r[4], dtype=np.float64).reshape(3, 3)
                inliers_idxi = np.fromstring(r[3], dtype=np.uint32).reshape(r[1], r[2])
                inl_maski[inliers_idxi[:, 0]] = True

            else:
                Fi = np.eye(3)

            image_ids2row[(img_id1, img_id2)] = (Fi, inl_maski)

        db.close()

        F = []
        inl_mask = []

        for img_n1, img_n2 in zip(image_name1, image_name2):
            img_id1, img_id2 = self.image_name2id[img_n1], self.image_name2id[img_n2]

            Fi, inl_maski = image_ids2row[(img_id1, img_id2)]

            F.append(Fi)
            inl_mask.append(inl_maski)

        return np.stack(F), np.stack(inl_mask)

    def get_qvec_tvec_and_inlier_mask(self, image_name1, image_name2, num_kp):
        db = COLMAPDatabase.connect(self.rec_db_path)

        image_ids2row = {}

        rows = db.get_qvec_tvec_and_inliers()

        for r in rows:
            img_id1, img_id2 = pair_id_to_image_ids(r[0])

            qveci = np.fromstring(r[4], dtype=np.float64)
            tveci = np.fromstring(r[5], dtype=np.float64)
            inliers_idxi = np.fromstring(r[3], dtype=np.uint32).reshape(r[1], r[2])

            inl_maski = np.zeros((num_kp), dtype=np.bool)
            inl_maski[inliers_idxi[:, 0]] = True

            image_ids2row[(img_id1, img_id2)] = (qveci, tveci, inl_maski)

        db.close()

        qvec = []
        tvec = []
        inl_mask = []

        for img_n1, img_n2 in zip(image_name1, image_name2):
            img_id1, img_id2 = self.image_name2id[img_n1], self.image_name2id[img_n2]

            qveci, tveci, inl_maski = image_ids2row[(img_id1, img_id2)]

            qvec.append(qveci)
            tvec.append(tveci)
            inl_mask.append(inl_maski)

        return np.stack(qvec), np.stack(tvec), np.stack(inl_mask)

    def delete_reconstruction(self):
        shutil.rmtree(self.rec_path)


"""
Legacy code
"""

# def bool2str(b):
#     return '1' if b else '0'

# def import_images_and_cameras(self, images, cameras):
    #     db = COLMAPDatabase.connect(self.rec_db_path)
    #
    #     for v in cameras.values():
    #         db.update_camera(v.id, v.model.model_id, v.width, v.height, v.params)
    #
    #     for v in images.values():
    #         db.update_image(v.id, v.name, v.camera_id, v.qvec, v.tvec)
    #
    #     db.commit()
    #     db.close()