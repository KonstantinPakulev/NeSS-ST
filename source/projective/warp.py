from source.projective.rbt import warp_image_rbt, warp_points_rbt
from source.projective.homography import warp_image_h, warp_points_h

from source.datasets.base.utils import RBTDataWrapper, HDataWrapper


def warp_image(image1, data, mode):
    if isinstance(data, RBTDataWrapper):
        return warp_image_rbt(image1, data, mode)

    elif isinstance(data, HDataWrapper):
        return warp_image_h(image1, data, mode)

    else:
        raise ValueError(f"Unknown data wrapper {data.__class__}")


def warp_points(points1, data):
    if isinstance(data, RBTDataWrapper):
        return warp_points_rbt(points1, data)

    elif isinstance(data, HDataWrapper):
        return warp_points_h(points1, data, 'pm')

    else:
        raise ValueError(f"Unknown data wrapper {data.__class__}")


# def warp_points_batch(kp1, kp2, batch, device):
#     kp1, kp2 = kp1.to(device), kp2.to(device)
#
#     image1, image2 = batch[du.IMAGE1], batch[du.IMAGE2]
#
#     if du.H1 in batch:
#         H12, H21 = batch[du.H1].to(device), batch[du.H2].to(device)
#
#         w_kp1, w_kp_mask1 = warp_points_h(kp1, image2.shape, H12)
#         w_kp2, w_kp_mask2 = warp_points_h(kp2, image1.shape, H21)
#
#     else:
#         scene_data1 = SceneDataWrapper.from_batch(batch, device)
#         scene_data2 = scene_data1.swap()
#
#         w_kp1, w_kp_mask1 = warp_points_rbt(kp1, scene_data1)
#         w_kp2, w_kp_mask2 = warp_points_rbt(kp2, scene_data2)
#
#     return w_kp1, w_kp2, w_kp_mask1, w_kp_mask2
#
#
# def warp_points_endpoint(batch, endpoint, device):
#     kp1, kp2 = endpoint[KP1], endpoint[KP2]
#
#     w_kp1, w_kp2, w_kp_mask1, w_kp_mask2 = warp_points_batch(kp1, kp2, batch, device)
#
#     endpoint[W_KP1] = w_kp1
#     endpoint[W_KP2] = w_kp2
#
#     endpoint[W_KP_MASK1] = w_kp_mask1
#     endpoint[W_KP_MASK2] = w_kp_mask2

# import source.datasets.base.utils as du
#
# from source.datasets.base.utils import RBTDataWrapper
# from source.projective.rbt import warp_points_rbt
# from source.projective.homography import warp_points_h