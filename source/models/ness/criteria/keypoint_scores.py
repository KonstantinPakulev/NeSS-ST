from source.utils.common_utils import get_eigen_values
from source.models.ness.criteria.homography import get_ww_patch_kp


def get_stability_score(image_gray, kp, h_data,
                        scaled_patch_size, patch_size,
                        num_samples,
                        k, nms_size,
                        base_detector):
    ww_patch_kp = get_ww_patch_kp(image_gray, kp, h_data,
                            scaled_patch_size, patch_size,
                            num_samples, k,
                            nms_size,
                            base_detector)

    kp_cov = (ww_patch_kp.unsqueeze(-1) @ ww_patch_kp.unsqueeze(-2)).sum(dim=0) / num_samples
    kp_ss = get_eigen_values(kp_cov).max(dim=-1)[0]

    return kp_ss


def get_repeatability_score(image_gray, kp, h_data,
                            scaled_patch_size, patch_size,
                            num_samples,
                            k, nms_size,
                            base_detector):
    ww_patch_kp = get_ww_patch_kp(image_gray, kp, h_data,
                            scaled_patch_size, patch_size,
                            num_samples, k,
                            nms_size,
                            base_detector)

    kp_rs = (ww_patch_kp.abs() <= 0.5).float().prod(dim=-1).sum(dim=0) / num_samples

    return kp_rs
