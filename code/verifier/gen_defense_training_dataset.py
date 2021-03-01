import json
import os

import cv2
import numpy as np
import torch

from config import Dataset_Path
from verifier.defense import LaneDefense, color_map_text, color_map

import matplotlib.pyplot as plt

import pickle as pkl

os.chdir("../../")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_train_data", "-gtd", action="store_true")
    parser.add_argument("--exp_name", "-name", default="original")
    parser.add_argument("--gt_file", default=os.path.join(Dataset_Path['Tusimple'],"label_data_0531.json"))
    parser.add_argument("--hnet_weight_path", '-hw', type=str, help="Path to hnet model weights", default="experiments_hnet/exp2/exp2_train.pth")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--neg_count", type=int, default=3)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--gen_samples", action="store_true")
    parser.add_argument("--set_type", default="val")
    parser.add_argument("--out_dir", default="defense/val_resize_sr1_ord2")
    args = parser.parse_args()

    im_count = 0
    if args.gen_train_data:


        gt_json, src_dir = LaneDefense.load_gt_data(args.gt_file)
        if args.limit > 0:
            gt_json = gt_json[:args.limit]
        output_data = []
        count = 0

        defense_gt = []

        for line in gt_json:

            img = LaneDefense.load_gt_img(line, src_dir)

            image_lanenet = cv2.resize(img, (512, 288), interpolation=cv2.INTER_AREA)


            lane_pts = []
            y_vals = np.array(line['h_samples'])


            os.makedirs(os.path.join(args.out_dir, "{}".format(im_count)), exist_ok=True)
            os.makedirs(os.path.join(args.out_dir, "{}/positive".format(im_count)), exist_ok=True)
            os.makedirs(os.path.join(args.out_dir, "{}/negative".format(im_count)), exist_ok=True)
            os.makedirs(os.path.join(args.out_dir, "{}/negative/reference/".format(im_count)), exist_ok=True)
            neg_curve_ref = []


            for ki in range(len(line['lanes'])):
                k = line['lanes'][ki]

                l = np.array(k)
                mask = l > 0
                if np.count_nonzero(mask) == 0:
                    continue

                fit_params = np.polyfit(y_vals[mask], l[mask], 3)
                ys_expand = np.arange(np.min(y_vals[mask]), np.max(y_vals[mask]), 0.1)
                xs_expand = np.polyval(fit_params, ys_expand)
                xs = xs_expand
                ys = ys_expand


                lane_pts.append(list(zip(xs, ys)))

            if len(lane_pts) == 0:
                continue

            color_count = 0
            for l_coords in lane_pts:
                try:
                    l_coords = np.array(l_coords)

                    l_coords[l_coords >= 1280] = 1279
                    tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
                    tmp_mask[np.int_(l_coords[:, 1]), np.int_(l_coords[:, 0])] = 255

                    if args.gen_samples:
                        color = color_map[color_count]

                        image_cpy = img.copy()
                        for j in l_coords:

                            cv2.circle(image_cpy, center=(int(j[0]), int(j[1])),
                                       radius=2,
                                                  color=(int(color[0]), int(color[1]), int(color[2])), thickness=2)



                        plt.imshow(cv2.cvtColor(image_cpy, cv2.COLOR_BGR2RGB))
                        plt.savefig(os.path.join(args.out_dir, "{}/{}_{}_orig.png".format(im_count, im_count, color_count)))

                    tmp_mask = cv2.resize(tmp_mask, (512, 288), interpolation=cv2.INTER_AREA)

                    nonzero_y = np.array(tmp_mask.nonzero()[0])
                    nonzero_x = np.array(tmp_mask.nonzero()[1])


                    p = np.polyfit(nonzero_y, nonzero_x, 3)


                    [ipm_image_height, ipm_image_width] = tmp_mask.shape

                    plot_y = np.linspace( np.min(nonzero_y), ipm_image_height, (ipm_image_height - 10))
                    fit_x = np.polyval(p, plot_y)

                    stabilized_lane = LaneDefense.get_stabilized_lane(nonzero_x, nonzero_y, image_lanenet, sample_rate=1)

                    np.save(os.path.join(args.out_dir, "{}/positive/{}_{}_stabilized_lane.npy".format(im_count, im_count, color_count)), cv2.cvtColor(stabilized_lane, cv2.COLOR_BGR2RGB))


                    for neg_count in range(args.neg_count):

                        perturbation_foundation = np.random.choice(np.arange(len(l_coords)), 2)
                        perturbation_foundation = l_coords[perturbation_foundation]

                        perturbation_y = np.append(perturbation_foundation[:,1], np.random.randint(150, 680, 3))
                        perturbation_x = np.append(perturbation_foundation[:,0], np.random.randint(100, 1150, 3))
                        fit_params = np.polyfit(perturbation_y, perturbation_x, 3)



                        ys_expand = np.arange(np.min(perturbation_y), 720, 0.1)
                        xs_expand = np.polyval(fit_params, ys_expand)

                        l_coords_random = list(zip(xs_expand, ys_expand))

                        l_coords_random = np.array(l_coords_random)

                        l_coords_random = np.clip(l_coords_random, 0, 1279)
                        tmp_mask_random = np.zeros(shape=(720, 1280), dtype=np.uint8)
                        tmp_mask_random[np.int_(np.clip(l_coords_random[:, 1], 0, 719)), np.int_(l_coords_random[:, 0])] = 255


                        tmp_mask_random = cv2.resize(tmp_mask_random, (512, 288), interpolation=cv2.INTER_AREA)
                        nonzero_y_random = np.array(tmp_mask_random.nonzero()[0])
                        nonzero_x_random = np.array(tmp_mask_random.nonzero()[1])

                        p_random = np.polyfit(nonzero_y_random, nonzero_x_random, 3)

                        [ipm_image_height_random, ipm_image_width_random] = tmp_mask_random.shape
                        plot_y_random = np.linspace(np.min(nonzero_y_random), ipm_image_height_random, (ipm_image_height_random - 10))

                        fit_x_random = np.polyval(p_random, plot_y_random)


                        stabilized_lane = LaneDefense.get_stabilized_lane(fit_x_random, plot_y_random, image_lanenet, sample_rate=1)
                        if stabilized_lane is not False and np.prod(stabilized_lane.shape) > 0 and np.count_nonzero(stabilized_lane) / np.prod(stabilized_lane.shape) > 0.8:
                            sl = plt.figure("sl")


                            np.save(os.path.join(args.out_dir,
                                "{}/negative/{}_{}_stabilized_lane.npy".format(im_count, im_count, color_count + neg_count*10)), cv2.cvtColor(stabilized_lane, cv2.COLOR_BGR2RGB))





                    color_count += 1



                except Exception as e:
                    print("lane processing error!")
                    print(e)


            with open(os.path.join(args.out_dir, "{}/negative/reference/negative_reference.pkl".format(im_count)),
                      "wb") as negative_reference:
                pkl.dump(neg_curve_ref, negative_reference)
            print(f"finished {im_count} negative reference")


            im_count += 1

            if im_count % 10 == 0:
                print(f'Completed {im_count} images.')

            line["defense_dir"] = os.path.join(args.out_dir, "{}/".format(im_count))

            plt.close(fig="all")

            continue

