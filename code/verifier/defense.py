import json
import numpy as np
import cv2
import os
import os.path as ops
import matplotlib.pyplot as plt
import math
from multiprocessing import Pool
import time

from config import Dataset_Path

color_map = [np.array([255, 0, 0]),
            np.array([0, 255, 0]),
            np.array([0, 0, 255]),
            np.array([125, 125, 0]),
            np.array([0, 125, 125]),
            np.array([125, 0, 125]),
            np.array([50, 100, 50]),
            np.array([100, 50, 100])]

color_map_text = ['blue', 'green', 'red', 'teal']
class LaneDefense(object):


    @staticmethod
    def rotate(image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated, M

    @staticmethod
    def get_stabilized_lane(xs, ys, img, offset=None, expand=True, derivative=None, set_pixels=False, only_index=False, sample_rate=0.5, ord=3):

        # print(len(ys))
        lane = []
        xs = np.array(xs)
        ys = np.array(ys)
        mask = xs>0 # remove out of lane points
        xs = xs[mask]
        ys = ys[mask]

        if len(xs) == 0:
            print("emtpy lane!")
            return False


        ys_orig = ys

        if offset is not None:
            ys = ys + offset

        # print("offset", len(ys))

        # increase samples
        if expand and derivative is None:
            fit_params = np.polyfit(ys, xs, ord)
            ys_expand = np.arange(np.min(ys_orig), np.max(ys_orig), sample_rate)
            xs_expand = np.polyval(fit_params, ys_expand)
            xs = xs_expand
            ys = ys_expand
            # print("xs_expand", len(xs))
            # print("ys_expand", len(ys))

        if derivative is None:
            angles = np.arctan(np.diff(xs)/np.diff(ys))
        else:
            angles = np.arctan(np.ones(derivative.shape)/derivative)

        times = []

        M = []
        e_point = []
        start_time = time.perf_counter_ns()
        times = []
        for a_i in range(len(angles)):
            a = angles[a_i]

            extracted_rect_setup = LaneDefense.extract_rect_rotate_selection(a, xs[a_i], ys[a_i], set_pixels, only_index)
            if set_pixels and extracted_rect_setup is False:
                print("0 derivative!")
                return False
            M.append(extracted_rect_setup[0])
            e_point.append(extracted_rect_setup[1])
            times.append(extracted_rect_setup[2])

        end_time = time.perf_counter_ns()
        # print("gen rotation matrix", end_time - start_time, np.sum(times))


        M = np.array(M)
        e_point = np.array(e_point)
        # print(extraction_pt_rotated.shape)


        if len(M):
            start_time = time.perf_counter_ns()
            extraction_pt_rotated = (M @ e_point).astype(int)
            end_time = time.perf_counter_ns()
            # print("multiply rotation matrix", end_time - start_time)

            extraction_pt_rotated[:, 1] = np.clip(extraction_pt_rotated[:, 1], 0, img.shape[0] - 1)
            extraction_pt_rotated[:, 0] = np.clip(extraction_pt_rotated[:, 0], 0, img.shape[1] - 1)

            if not set_pixels and not only_index:
                lane = img[extraction_pt_rotated[:, 1], extraction_pt_rotated[:, 0]]

            else:
                for r in extraction_pt_rotated:

                    row = r[1]
                    col = r[0]

                    print(row.shape)

                    if set_pixels:
                        img[row, col] = 255

                    if only_index:
                        extracted_rect = row, col
                        lane.append(extracted_rect)





        # print(f"extraction_time {np.sum(times)}")

        if only_index:
            return lane
        if not set_pixels:
            return np.array(lane)
        else:
            return img

    @staticmethod
    def extract_rect_rotate_img(img, a, idx, xs, ys):
        img_rotated, M = LaneDefense.rotate(img, np.rad2deg(-a), center=(xs[idx], ys[idx]))

        extraction_pt = np.array([[xs[idx], ys[idx], 1],
                                  [xs[idx + 1], ys[idx + 1], 1]]).T
        extraction_pt_rotated = M @ extraction_pt
        extraction_center = extraction_pt_rotated.T[0]
        extraction_height = extraction_pt_rotated.T[1][1] - extraction_center[1]
        extracted_rect = LaneDefense.extract_rect(extraction_center.astype(int), img_rotated,
                                                  height=int(extraction_height))
        return extracted_rect

    @staticmethod
    def extract_rect(extraction_pt, img, width=20, height=50):
        return img[extraction_pt[1]:extraction_pt[1]+height, int(extraction_pt[0]-width/2):int(extraction_pt[0]+width/2)]


    @staticmethod
    def extract_rect_rotate_selection(a, x, y, set_pixels=False, only_index=False):
        if set_pixels and np.abs(np.rad2deg(a)) < 1:
            return False
        xs_int = int(x)
        if set_pixels:
            x_grid = np.random.randint(xs_int - 6, xs_int + 6, 50, dtype=np.float32)
        else:
            x_grid = np.arange(xs_int-20, xs_int + 20, 1, dtype=np.float32)



        extraction_loc = [x_grid,
                          [y] * x_grid.shape[0],
                        [1] * x_grid.shape[0]]

        start_time = time.perf_counter_ns()

        M = cv2.getRotationMatrix2D((xs_int, y), np.rad2deg(a), 1.0)

        # alpha = np.cos(a)
        # beta = np.sin(a)
        # M = [
        #     [alpha, beta, (1-alpha) * xs_int + beta * y],
        #     [-beta, alpha, beta * xs_int + (1-alpha) * y]
        # ]

        end_time = time.perf_counter_ns()

        return (M, extraction_loc, end_time - start_time)

    @staticmethod
    def load_gt_lane(gt_file, selected_idx=0):
        json_gt, src_dir = LaneDefense.load_gt_data(gt_file)
        src_dir = ops.split(gt_file)[0]
        selected_img = json_gt[selected_idx]
        return LaneDefense.load_gt_img(json_gt, src_dir), selected_img

    @staticmethod
    def load_gt_data(gt_file):
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        src_dir = ops.split(gt_file)[0]
        return json_gt, src_dir

    @staticmethod
    def load_gt_img(gt_dict, src_dir):
        selected_img = gt_dict
        image_path = ops.join(src_dir, selected_img['raw_file'])
        img = cv2.imread(image_path)
        return img





if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--single_index", "-si", type=int, default=-1)
    parser.add_argument("--gen_train_data", "-gtd", action="store_true")
    parser.add_argument("--exp_name", "-name", default="original")
    parser.add_argument("--gt_file", default=os.path.join(Dataset_Path['Tusimple'],"label_data_0313.json"))
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    out_dir = ops.join("demo", args.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    if args.single_index >= 0:
        image, s_img = LaneDefense.load_gt_lane(args.gt_file, selected_idx=args.index)

        if args.show:
            image_cpy = image.copy()
            for l in range(len(s_img['lanes'])):
                lane = s_img['lanes'][l]
                color = color_map[l]
                perturb = np.linspace(50, 0, np.count_nonzero(np.array(lane)>0))
                idx = 0



                for j in range(len(lane)):

                    if lane[j]>0:

                        cv2.circle(image_cpy, center=(int(lane[j]), int(s_img['h_samples'][j])),
                                   radius=2,
                                              color=(int(color[0]), int(color[1]), int(color[2])), thickness=2)
                        # cv2.circle(image, center=(int(lane[j] + np.random.randint(0, 50)), int(s_img['h_samples'][j])), radius=3,
                        #            color=(int(color[0]), int(color[1]), int(color[2])), thickness=2)
                        # cv2.circle(image_cpy, center=(int(lane[j] + perturb[idx]), int(s_img['h_samples'][j])),
                        #            radius=3,
                        #            color=(int(color[0]), int(color[1]), int(color[2])), thickness=2)
                        idx += 1

        g_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        count = 0
        for lane in s_img['lanes']:
            isolated_lane = LaneDefense.get_stabilized_lane(lane, s_img['h_samples'], g_img)
            # print(isolated_lane.shape)
            plt.figure("lane {}".format(count))
            plt.imshow(isolated_lane)
            plt.savefig(ops.join(out_dir, "{}.png".format(count)))
            count += 1
        plt.figure("full img")
        plt.imshow(g_img)
        plt.savefig(ops.join(out_dir,"full.png".format(count)))
        plt.show()

    if args.gen_train_data:
        gt_json, src_dir = LaneDefense.load_gt_data(args.gt_file)
        if args.limit > 0:
            gt_json = gt_json[:args.limit]
        output_data = []
        count = 0
        for line in gt_json:
            img = LaneDefense.load_gt_img(line, src_dir)

            if args.show:
                image_cpy = img.copy()
                image_cpy_gt = img.copy()
                for l in range(len(line['lanes'])):
                    lane = line['lanes'][l]
                    color = color_map[l]

                    idx = 0

                    xs = np.array(lane)
                    ys = np.array(line['h_samples'])
                    mask = xs > 0  # remove out of lane points
                    xs = xs[mask]
                    ys = ys[mask]

                    if len(xs) == 0:
                        continue

                    fit_params = np.polyfit(ys, xs, 3)
                    ys_expand = np.arange(np.min(ys), np.max(ys), 0.5)
                    xs_expand = np.polyval(fit_params, ys_expand)
                    xs = xs_expand
                    ys = ys_expand

                    for j in range(len(xs)):
                        cv2.circle(image_cpy, center=(int(xs[j]), int(ys[j])),
                                   radius=2,
                                   color=(int(color[0]), int(color[1]), int(color[2])), thickness=2)

                    for j in range(len(lane)):
                        if lane[j] > 0:
                            cv2.circle(image_cpy_gt, center=(int(lane[j]), int(line['h_samples'][j])),
                               radius=5,
                               color=(int(color[0]), int(color[1]), int(color[2])), thickness=5)

                alpha = 0.6
                image_overlay = cv2.addWeighted(image_cpy, alpha, img, 1 - alpha, gamma=0)
                alpha_gt = 0.2
                image_overlay = cv2.addWeighted(image_cpy_gt, alpha_gt, image_overlay, 1 - alpha_gt, gamma=0)
                image_overlay = cv2.cvtColor(image_overlay, cv2.COLOR_BGR2RGB)
                all_fig = plt.figure("highlighted lanes")
                plt.title("all highlighted lanes")
                plt.imshow(image_overlay)
                plt.savefig("defense/identify_lanes/{}_all_lanes.png".format(count))
                plt.close(all_fig)




            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            y_vals = line['h_samples']
            lane_count = 0
            for lane in line['lanes']:
                isolated_lane = LaneDefense.get_stabilized_lane(lane, y_vals, img)
                if isolated_lane is False:
                    # print(lane)
                    continue
                output_data.append({
                    "label": 1,
                    "data": isolated_lane.tolist()
                })

                # print(output_data[-1])


                if args.show:
                    lane_title = "lane {}:{}".format(lane_count, color_map_text[lane_count])
                    l_fig = plt.figure(lane_title)
                    plt.title(lane_title)
                    isolated_lane_rgb = cv2.cvtColor(isolated_lane, cv2.COLOR_BGR2RGB)
                    plt.imshow(isolated_lane)

                    plt.savefig("defense/identify_lanes/{}_lane_{}.png".format(count, color_map_text[lane_count]))
                    plt.close(l_fig)
                    lane_count += 1

                # print(isolated_lane.shape[0])

                isolated_pertubation = LaneDefense.get_stabilized_lane(lane, y_vals, img, offset=np.linspace(50, 0, np.count_nonzero(np.array(lane)>0)))
                output_data.append({
                    "label": 0,
                    "data": isolated_pertubation.tolist()
                })

                # print(isolated_pertubation.shape[0])

            # plt.show()

            count += 1
            if count % 100 == 0:
                print("completed {}".format(count))

        with open("defense/color_{}.txt".format(args.limit), "w+", encoding='utf-8') as f:
            for d in output_data:
                json.dump(d, f)
                f.write("\n")






