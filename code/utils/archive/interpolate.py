from numpy import RankWarning

from utils.archive.H_net import *
import cv2
import warnings

import matplotlib.pyplot as plt
fixed_transformation = get_H_net_transformation()

# preconfigured ipm from
# https://github.com/MaybeShewill-CV/lanenet-lane-detection/blob/9154c1497291276f03fb70a33149b152aa74cda3/lanenet_model/lanenet_postprocess.py
def _load_remap_matrix():
    """
    :return:
    """
    fs = cv2.FileStorage(_ipm_remap_file_path, cv2.FILE_STORAGE_READ)

    remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
    remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

    ret = {
        'remap_to_ipm_x': remap_to_ipm_x,
        'remap_to_ipm_y': remap_to_ipm_y,
    }

    fs.release()

    return ret

_ipm_remap_file_path = "./utils/tusimple_ipm_remap.yml"

remap_file_load_ret = _load_remap_matrix()
_remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
_remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')


def get_lane_interpolations_tusimple(lane_seg_img, source_image):
    num_lanes = len(np.unique(lane_seg_img))
    # print(num_lanes)
    # print(num_lanes)
    # print(np.unique(lane_seg_img))
    points_per_lane = get_points_per_lane(lane_seg_img, num_lanes)

    src_lane_pts = []
    src_lane_pts_x = []
    src_lane_pts_y = []
    # lane_polynomials = [0 for i in range(num_lanes)]
    for lane_idx in range(num_lanes):
        if lane_idx == 0:
            continue
        points = np.array(points_per_lane[lane_idx])
        # print(np.max(points, axis=0))
        tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
        tmp_mask[tuple((np.int_(points[:, 0] * 720 / 288), np.int_(points[:, 1] * 1280 / 512)))] = 255
        # print(tmp_mask.nonzero())
        # print(_remap_to_ipm_x.shape)
        # print(_remap_to_ipm_y.shape)
        # plt.imshow(tmp_mask)
        # plt.show()
        tmp_ipm_mask = cv2.remap(
            tmp_mask,
            _remap_to_ipm_x,
            _remap_to_ipm_y,
            interpolation=cv2.INTER_NEAREST
        )
        # print(tmp_ipm_mask.nonzero())
        # plt.imshow(tmp_ipm_mask)
        # plt.show()
        nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
        nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

        # print(nonzero_x)
        # print(nonzero_y)
        debug = False
        if len(nonzero_x) > 0:

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    p = np.polyfit(nonzero_y, nonzero_x, 2)
                except RankWarning as e:
                    print('error found:', e)
                    debug = True
            # lane_polynomials[lane_idx] = p

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
            fit_x = p[0] * plot_y ** 2 + p[1] * plot_y + p[2]
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

            lane_pts = []
            lane_pts_x = []
            lane_pts_y = []
            for index in range(0, plot_y.shape[0], 5):
                src_x = _remap_to_ipm_x[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                if src_x <= 0:
                    continue
                src_y = _remap_to_ipm_y[
                    int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                src_y = src_y if src_y > 0 else 0

                lane_pts.append([src_x, src_y])
                lane_pts_x.append(src_x)
                lane_pts_y.append(src_y)
        else:
            lane_pts = [[-1,-1]]
            lane_pts_x = [-1]
            lane_pts_y = [-1]

        src_lane_pts.append(lane_pts)
        src_lane_pts_x.append(lane_pts_x)
        src_lane_pts_y.append(lane_pts_y)

    # lane_polynomials = np.array(lane_polynomials)

    if source_image is not None:
        source_image_width = source_image.shape[1]
    else:
        source_image_width = 1280

    all_coords = []
    for index, single_lane_pts in enumerate(src_lane_pts):
        single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
        single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]

        coords = []
        # for plot_y in np.linspace(start_plot_y, end_plot_y, step):
        for plot_y in [720 - 10 * i for i in range(1, 56 + 1)]:
            diff = single_lane_pt_y - plot_y
            fake_diff_bigger_than_zero = diff.copy()
            fake_diff_smaller_than_zero = diff.copy()
            fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf')
            fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf')
            idx_low = np.argmax(fake_diff_smaller_than_zero)
            idx_high = np.argmin(fake_diff_bigger_than_zero)

            previous_src_pt_x = single_lane_pt_x[idx_low]
            previous_src_pt_y = single_lane_pt_y[idx_low]
            last_src_pt_x = single_lane_pt_x[idx_high]
            last_src_pt_y = single_lane_pt_y[idx_high]

            if previous_src_pt_y < 150 or last_src_pt_y < 150 or \
                    fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                    fake_diff_bigger_than_zero[idx_high] == float('inf'):
                coords.append((-1, int(plot_y)))
                continue

            interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                      abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                     (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
            interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                      abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                     (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

            if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                coords.append((-1, int(interpolation_src_pt_y)))
                continue

            lane_color = color[index].tolist()
            if source_image is not None:
                cv2.circle(source_image, (int(interpolation_src_pt_x),
                                          int(interpolation_src_pt_y)), 5, lane_color, -1)
            coords.append((int(interpolation_src_pt_x),
                           int(interpolation_src_pt_y)))
        all_coords.append(coords)

    ret = {
        'source_image': source_image,
        'tu_simple_coords': all_coords
    }
    if debug:
        pass

    # print(len(lane_polynomials))
    # print(len(lane_pts_x))
    return ret


# gets polynomials from points per lane objects
def get_lane_interpolations_with_pts(lane_points, transformation, n=3):
    y_coord_proj, lane_polynomial = interpolate(lane_points, transformation, n)
    x_coord_proj = np.polyval(lane_polynomial, y_coord_proj)

    proj_matrix = np.vstack([x_coord_proj, y_coord_proj, np.ones(y_coord_proj.shape)])

    ret = np.linalg.inv(transformation) @ transformation @ lane_points
    plt.scatter(ret[0,:], ret[1,:])
    plt.show()
    return ret


# Interpolates x, y coordinates into an n-degree polynomial in the transformation space
def interpolate(points, transformation, n):
    """
    Performs least squares polynomical regression on the lane points
    This needs to be done for each lane to get a polynomial in the transformed img
    It uses np.polyfit which uses a Least squares polynomial fit.
    """

    # points = np.array(points)
    # points_h = np.hstack((points, np.ones((len(points), 1)))).T
    # print(points)
    plt.scatter(points[0,:], points[1,:])
    points_proj = np.dot(transformation, points)
    # print(points_proj)
    x_coord_proj = points_proj[0,:]
    y_coord_proj = points_proj[1,:]
    plt.scatter(x_coord_proj, y_coord_proj)

    p = np.polyfit(y_coord_proj, x_coord_proj, n)

    return y_coord_proj, p