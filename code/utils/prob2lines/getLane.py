import numpy as np
import torch

from defense import LaneDefense
from utils.archive.interpolate import get_lane_interpolations_tusimple
from numpy import RankWarning
import warnings

import matplotlib.pyplot as plt

from utils.transforms import Compose, ToTensor, Normalize, Resize

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)


def polyfit2coords_tusimple(lane_pred, crop_h=0, resize_shape=None, y_px_gap=20, pts=None, ord=2):
    if resize_shape is None:
        resize_shape = lane_pred.shape
        crop_h = 0
    h, w = lane_pred.shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)



    for i in [idx for idx in np.unique(lane_pred) if idx!=0]:
        ys_pred, xs_pred = np.where(lane_pred==i)

        poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
        ys = np.array([h-y_px_gap/(H-crop_h)*h*i for i in range(1, pts+1)])
        xs = np.polyval(poly_params, ys)

        y_min, y_max = np.min(ys_pred), np.max(ys_pred)
        coordinates.append([[int(x/w*W) if x>=0 and x<w and ys[i]>=y_min and ys[i]<=y_max else -1,
                             H-y_px_gap*(i+1)] for (x, i) in zip(xs, range(pts))])

    return coordinates

def polyfit2coords_tusimple_with_bounded_classifier(lane_pred, img, classifier, classifier_resize_shape=(),
                                            sample_rate=1,
                                            crop_h=0, resize_shape=None, y_px_gap=20, pts=None, ord=3):
    if resize_shape is None:
        resize_shape = lane_pred.shape
        crop_h = 0
    h, w = lane_pred.shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)


    transform_x = Compose(Resize(classifier_resize_shape), ToTensor())



    flagged = 0
    for i in [idx for idx in np.unique(lane_pred) if idx!=0]:
        ys_pred, xs_pred = np.where(lane_pred==i)

        debug = False
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                stabilized_lane = LaneDefense.get_stabilized_lane(xs_pred, ys_pred, img, sample_rate=sample_rate, ord=ord)
            except RankWarning as e:
                debug = True

        stabilized_lane = LaneDefense.get_stabilized_lane(xs_pred, ys_pred, img, sample_rate=sample_rate, ord=ord)
        if len(stabilized_lane) > 0:

            if debug:
                plt.imshow(stabilized_lane)
                plt.show()

            # plt.imshow(stabilized_lane)
            class_logit = classifier(
                transform_x({"img": stabilized_lane})["img"].cuda().unsqueeze(0)).detach().cpu().numpy()
            classification = 1/(1 + np.exp(-class_logit)) > 0.78


            # classification = output_sum > 2
            # print(output_list)
            # print(classification)
            # print("-----------")


        else:
            classification = True

        if classification:
            poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
            ys = np.array([h - y_px_gap / (H - crop_h) * h * i for i in range(1, pts + 1)])
            xs = np.polyval(poly_params, ys)

            y_min, y_max = np.min(ys_pred), np.max(ys_pred)
            coordinates.append([[int(x / w * W) if x >= 0 and x < w and ys[i] >= y_min and ys[i] <= y_max else -1,
                                 H - y_px_gap * (i + 1)] for (x, i) in zip(xs, range(pts))])
        else:
            flagged += 1

    return coordinates



def polyfit2coords_tusimple_with_classifier2(lane_pred, img, classifier,
                                            sample_rate=1, patch_height=60, min_patch=3,
                                            crop_h=0, resize_shape=None, y_px_gap=20, pts=None, ord=3):
    if resize_shape is None:
        resize_shape = lane_pred.shape
        crop_h = 0
    h, w = lane_pred.shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)


    flagged = 0
    for i in [idx for idx in np.unique(lane_pred) if idx!=0]:
        ys_pred, xs_pred = np.where(lane_pred==i)

        debug = False
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                stabilized_lane = LaneDefense.get_stabilized_lane(xs_pred, ys_pred, img, sample_rate=sample_rate, ord=ord)
            except RankWarning as e:
                debug = True

        stabilized_lane = LaneDefense.get_stabilized_lane(xs_pred, ys_pred, img, sample_rate=sample_rate, ord=ord)
        if len(stabilized_lane) > 0:

            if debug:
                plt.imshow(stabilized_lane)
                plt.show()

            stabilized_orig_tensor = torch.from_numpy(stabilized_lane).permute(2, 0, 1)
            stabilized_orig_tensor = stabilized_orig_tensor.cuda()

            if stabilized_orig_tensor.shape[1] < patch_height * min_patch:
                stabilized_orig_tensor = torch.nn.AdaptiveAvgPool2d(
                    (patch_height * min_patch, stabilized_orig_tensor.shape[2]))(stabilized_orig_tensor)

            output_list = []
            for _ in range(9):
                # print(patch_height, stabilized_orig_tensor.shape[1])
                try:
                    rand_idx = np.random.randint(patch_height, stabilized_orig_tensor.shape[1])
                except:
                    continue
                patch = stabilized_orig_tensor[:, rand_idx - patch_height: rand_idx, :]
                outputs = classifier(patch.unsqueeze(0))
                output_list.append(outputs[0].detach().cpu().numpy() > 0)

            output_list = np.array(output_list).flatten()

            output_sum = np.sum(output_list)

            classification = output_sum > (output_list.shape[0] - output_sum)

            # classification = output_sum > 2
            # print(output_list)
            # print(classification)
            # print("-----------")


        else:
            classification = True

        if classification:
            poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
            ys = np.array([h - y_px_gap / (H - crop_h) * h * i for i in range(1, pts + 1)])
            xs = np.polyval(poly_params, ys)

            y_min, y_max = np.min(ys_pred), np.max(ys_pred)
            coordinates.append([[int(x / w * W) if x >= 0 and x < w and ys[i] >= y_min and ys[i] <= y_max else -1,
                                 H - y_px_gap * (i + 1)] for (x, i) in zip(xs, range(pts))])
        else:
            flagged += 1

    return coordinates


def polyfit2coords_tusimple_with_classifier(lane_coordinate, cluster_index, labels, img, classifier,
                                            sample_rate=1, patch_height=60, min_patch=3,
                                            crop_h=0, resize_shape=None, y_px_gap=20, pts=None, ord=2):
    if resize_shape is None:
        resize_shape = img.shape
        crop_h = 0
    h, w = img.shape[:2]
    H, W = resize_shape[:2]
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)



    # for i in [idx for idx in np.unique(lane_pred) if idx!=0]:
    #     ys_pred, xs_pred = np.where(lane_pred==i)


    flagged = 0
    for i in cluster_index:
        idx_l = np.where(labels == i)
        coord = lane_coordinate[idx_l]

        lane_pts = coord
        # print(lane_pts)
        xs_pred, ys_pred = lane_pts[:, 1], lane_pts[:, 0]


        debug = False
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                stabilized_lane = LaneDefense.get_stabilized_lane(xs_pred, ys_pred, img, sample_rate=sample_rate)
            except RankWarning as e:
                debug = True


        stabilized_lane = LaneDefense.get_stabilized_lane(xs_pred, ys_pred, img, sample_rate=sample_rate)
        if len(stabilized_lane) == 0:
            continue

        if debug:
            plt.imshow(stabilized_lane)
            plt.show()

        stabilized_orig_tensor = torch.from_numpy(stabilized_lane).permute(2, 0, 1)
        stabilized_orig_tensor = stabilized_orig_tensor.cuda()

        if stabilized_orig_tensor.shape[1] < patch_height * min_patch:
            stabilized_orig_tensor = torch.nn.AdaptiveAvgPool2d(
                (patch_height * min_patch, stabilized_orig_tensor.shape[2]))(stabilized_orig_tensor)

        output_list = []
        for _ in range(9):
            # print(patch_height, stabilized_orig_tensor.shape[1])
            try:
                rand_idx = np.random.randint(patch_height, stabilized_orig_tensor.shape[1])
            except:
                continue
            patch = stabilized_orig_tensor[:, rand_idx - patch_height: rand_idx, :]
            outputs = classifier(patch.unsqueeze(0))
            output_list.append(outputs[0].detach().cpu().numpy() > 0)

        output_list = np.array(output_list).flatten()

        output_sum = np.sum(output_list)

        classification = output_sum+2 > (output_list.shape[0] - output_sum)

        # print(output_list)
        # print(classification)
        # print("-----------")

        if classification:
            poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
            ys = np.array([h - y_px_gap / (H - crop_h) * h * i for i in range(1, pts + 1)])
            xs = np.polyval(poly_params, ys)

            y_min, y_max = np.min(ys_pred), np.max(ys_pred)
            coordinates.append([[int(x/w*W) if x>=0 and x<w and ys[i]>=y_min and ys[i]<=y_max else -1,
                                 H-y_px_gap*(i+1)] for (x, i) in zip(xs, range(pts))])
        else:
            flagged += 1

    return coordinates, flagged




def polyfit2coords_tusimple_hnet(lane_coords, orig_shape, resize_shape, crop_h=0, y_px_gap=20, pts=None, ord=3, catch=False):

    debug = False
    h, w = orig_shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)


    # print(lane_coords)
    for i in range(len(lane_coords)):
        lane_pts = lane_coords[i]
        xs_pred, ys_pred = lane_pts[0, :], lane_pts[1, :]

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
            except RankWarning as e:
                debug = True
        poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
        ys = np.array([h-y_px_gap/(H-crop_h)*h*i for i in range(1, pts+1)])
        xs = np.polyval(poly_params, ys)

        y_min, y_max = np.min(ys_pred), np.max(ys_pred)
        # if catch:
        #     print("xs", xs)
        #     print("ys", ys)
        #     print("xs_pred", xs_pred)
        #     print("ys_pred", ys_pred)
        #     print([[int(x/w*W) if x>=0 and x<w and ys[i]>=y_min and ys[i]<=y_max else -1,
        #                      H-y_px_gap*(i+1)] for (x, i) in zip(xs, range(pts))])
        cleaned_lane = [[int(x/w*W) if x>=0 and x<w and ys[i]>=y_min and ys[i]<=y_max else -1,
                             H-y_px_gap*(i+1)] for (x, i) in zip(xs, range(pts))]
        cleaned_lane_np = np.array(cleaned_lane)
        print_sep = False
        if np.count_nonzero(cleaned_lane_np[:, 0] < 0) < cleaned_lane_np.shape[0]:
            coordinates.append(cleaned_lane)
        else:
            print("empty lane!")
            print_sep = True
        # if catch:
        #     print("caught!")
        #     print(cleaned_lane)
        if debug:
            print("polyfit warning!")
            print(cleaned_lane)
        if catch or debug or print_sep:
            print("-------------")
    return coordinates, debug


def polyfit2coords_tusimple_ipm(lane_pred, crop_h=0, resize_shape=None, y_px_gap=20, pts=None, ord=2):
    if resize_shape is None:
        resize_shape = lane_pred.shape
        crop_h = 0
    h, w = lane_pred.shape
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)

    p, xs_ipm, ys_ipm = get_lane_interpolations_tusimple(lane_pred)
    for i in range(len(np.unique(lane_pred))-1):
        # print(len(np.unique(lane_pred)))
        # print(len(xs_ipm))
        ys_pred, xs_pred = ys_ipm[i], xs_ipm[i]

        poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
        ys = np.array([h - y_px_gap / (H - crop_h) * h * i for i in range(1, pts + 1)])
        xs = np.polyval(poly_params, ys)

        y_min, y_max = np.min(ys_pred), np.max(ys_pred)
        coordinates.append([[int(x / w * W) if x >= 0 and x < w and ys[i] >= y_min and ys[i] <= y_max else -1,
                             H - y_px_gap * (i + 1)] for (x, i) in zip(xs, range(pts))])

    return coordinates

def polyfit2coords_tusimple_ipm_postprocessed(lane_coords, crop_h=0, resize_shape=None, y_px_gap=20, pts=None, ord=2):
    if resize_shape is None:
        resize_shape = (720, 1280)
        crop_h = 0
    h, w = (720, 1280)
    H, W = resize_shape
    coordinates = []

    if pts is None:
        pts = round(H / 2 / y_px_gap)

    for i in range(len(lane_coords)):
        # print(len(np.unique(lane_pred)))
        # print(len(xs_ipm))
        lane = np.array(lane_coords[i])
        ys_pred, xs_pred = lane[:,0], lane[:,1]

        poly_params = np.polyfit(ys_pred, xs_pred, deg=ord)
        ys = np.array([h - y_px_gap / (H - crop_h) * h * i for i in range(1, pts + 1)])
        xs = np.polyval(poly_params, ys)

        y_min, y_max = np.min(ys_pred), np.max(ys_pred)
        coordinates.append([[int(x / w * W) if x >= 0 and x < w and ys[i] >= y_min and ys[i] <= y_max else -1,
                             H - y_px_gap * (i + 1)] for (x, i) in zip(xs, range(pts))])

    return coordinates

