import json
import shutil
import time
from datetime import datetime
import pickle as pkl

import sklearn
from torch import nn
from tqdm import tqdm
import torchvision.transforms.functional as F

from verifier.defense import LaneDefense


from verifier.defense_nn_models import DefenseMLP, DefenseLinear, DefenseMaxPool
from attack.lanenet_attack import calc_iou
from lane_proposal.model import LaneNet
from utils.postprocess import LaneNetCluster
from utils.transforms import *
import os

import matplotlib.pyplot as plt

config_files = []
config_files.append(r"defense\configs\exp115a3000e303_e2e_2021-02-14__14-53-51.config")



if __name__=='__main__':

    for config_file in config_files:
        base_dir = "defense"


        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                mean = config["mean"]
                std = config["std"]
                resize_shape = tuple(config["resize_shape"])
                kernel_size = config["kernel_size"]
                exp_num = config["exp_num"]
                hidden_layer1 = config["hidden_layer1"]
                hidden_layer2 = config["hidden_layer2"]
                model_type = config["model_type"]
                additional_modifiers = config["additional_modifiers"]
                noise_sd = config["noise_sd"]
                exp = config["exp"]
                trained_epoch = config.get("trained_epoch", 86)
                adv_train = config.get("adv_train", False)
                adv_exp = config.get("adv_exp", 404)
                adv_eps = config.get('adv_eps', 5/255)
                adv_iter = config.get('adv_iter', 3)
                e2e_expnum = config.get('e2e_expnum', "MANUAL")
                e2e_out_dir = config.get('e2e_out_dir')
                sample_rate = config.get("sample_rate", 0.5)
                outerloop = config.get("outerloop", config.get("e2e_outerloop", True))
                e2e_save_name = config.get("e2e_save_name")
                e2e_lanenet_dset = config.get("lanenet_dset", config.get("e2e_lanenet_dset", "val"))
                full_exp = config.get("full_exp", exp)
        else:
            raise AssertionError("config file doesn't exist!")


        if full_exp:
            exp = full_exp

        if f"e{e2e_expnum}" not in exp:
            exp += f"e{e2e_expnum}"

        config["e2e_supplied_config"] = config_file
        config["e2e_eval_classifier_attack"] = post_classifier = True
        config["e2e_polyfit_order"] = order = 3


        config["e2e_manual_epoch"] = manual_epoch = 1
        config["e2e_manual_data_dir"] = manual_data_dir = "defense\\attack\\e2e\\exp115a900e1000"


        config["e2e_threshold"] =  e2e_threshold = 0.313
        config["e2e_show_results"] = show_results = False

        config["test_notes"] = ""

        save_suffix = ".pth"


        if adv_train:
            base_dir = os.path.join(base_dir, "attack")
            save_suffix = "_advtrain.pth"


        save_name = e2e_save_name if e2e_save_name else os.path.join(base_dir, "saved_models", full_exp, str(manual_epoch) + save_suffix)
        save_dict = torch.load(save_name)
        device = torch.device("cuda:0")

        config["e2e_eval_save_name"] = save_name

        if model_type == "conv":
            model = DefenseMLP(kernel_size=kernel_size, input_shape=resize_shape).to(device)
        elif model_type == "conv_maxpool":
            model = DefenseMaxPool(kernel_size=kernel_size, input_shape=resize_shape).to(device)
        else:
            model = DefenseLinear(3 * np.prod(resize_shape), hidden_layer1, hidden_layer2).to(device)

        model.load_state_dict(save_dict['net'])

        model.eval()


        exp_dir = r"C:\Users\henry\Dropbox\sp20\backup\henry\PycharmProjects\tmp\lane-detection\experiments/exp0"
        exp_name = exp_dir.split('/')[-1]
        with open(os.path.join(exp_dir, "cfg.json")) as f:
            exp_cfg = json.load(f)

        net = LaneNet(pretrained=True, **exp_cfg['model'])
        net = net.to(device)
        net.eval()


        if torch.cuda.is_available():
            save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth'))
        else:
            save_dict = torch.load(os.path.join(exp_dir, exp_dir.split('/')[-1] + '.pth'), map_location=torch.device('cpu'))
        if isinstance(net, torch.nn.DataParallel):
            net.module.load_state_dict(save_dict['net'])
        else:
            net.load_state_dict(save_dict['net'])


        class Lane_C(nn.Module):
            def __init__(
                    self,
                    base_model
            ):
                super(Lane_C, self).__init__()
                self.base = base_model

            def forward(self, img):
                img = torch.stack([F.normalize(i, mean, std) for i in img])
                x = self.base(img)  # x is a dict
                bmap = x
                return bmap


        att_model = Lane_C(net).to(device)
        att_model.eval()

        class Lane_Resize(nn.Module):
            def __init__(
                    self,
                    base_model
            ):
                super(Lane_Resize, self).__init__()
                self.base = base_model

            def forward(self, img):
                img = torch.stack([F.normalize(i, mean, std) for i in img])
                img = nn.AdaptiveAvgPool2d(resize_shape[::-1])(img)
                x = self.base(img)  # x is a dict
                bmap = x
                return bmap


        model = Lane_Resize(model)

        transform_lanenet = Compose(ToTensor())
        transform_x = Compose(ToTensor())

        cluster = LaneNetCluster()



        data_dir = e2e_out_dir if e2e_out_dir else manual_data_dir

        eval_dir = os.path.join(base_dir, "logs", exp, "e2e", os.path.basename(data_dir))

        eval_dir = os.path.join(eval_dir, e2e_lanenet_dset)

        config["e2e_eval_dir"] = eval_dir


        os.makedirs(eval_dir, exist_ok=True)


        os.makedirs(os.path.join(eval_dir, "negative"), exist_ok=True)
        os.makedirs(os.path.join(eval_dir, "positive"), exist_ok=True)
        full_dir = e2e_out_dir if e2e_out_dir else manual_data_dir

        config["e2e_eval_full_dir"] = full_dir
        # full_dir = r"C:\Users\henry\Dropbox\sp20\backup\henry\PycharmProjects\tmp\lane-detection\defense\attack\e2e\" + data_dir
        error_count = 0
        true_count = 0
        total_true = 0

        out_file_time = time.strftime("%Y-%m-%d__%H-%M-%S")
        out_file_name =  f"{exp}_{'post' if post_classifier else 'pre'}_{out_file_time}.log"
        out_iou_name =  f"{exp}_{'post' if post_classifier else 'pre'}_{out_file_time}_iou.pkl"
        iou_list = []
        iou_classification = []

        roc_gt = []
        roc_logit = []
        roc_iou = []
        roc_iou_logits = []

        eval_out_dir = "eval5"
        with open(os.path.join("defense", eval_out_dir, out_file_name), 'a+') as log_file:
            print(json.dumps(config, indent=4, sort_keys=True), file=log_file)
            print(json.dumps(config, indent=4, sort_keys=True))
            print("loading examples from", full_dir)
            print("loading examples from", full_dir, file=log_file)
            with torch.no_grad():
                old_system = len(os.listdir(full_dir)) // 8
                if os.path.exists(os.path.join(full_dir, "adv")):
                    new_system = len(os.listdir(os.path.join(full_dir, "adv")))
                else:
                    new_system = 0
                candidate_count = max(old_system, new_system)
                for idx in tqdm(range(candidate_count)):
                    if "outerloop" in data_dir or outerloop:
                        ol_range = 4 if not outerloop or isinstance(outerloop, bool) else outerloop
                        suffix = True
                    else:
                        ol_range = 1
                        suffix = False
                    ol_range = 1
                    for ol in range(ol_range):
                        if suffix:
                            suffix = f"_{ol}"
                        else:
                            suffix = ""

                        idx_specific_error = 0
                        idx_specific_true = 0
                        idx_specific_true_total = 0


                        if post_classifier:
                            img_folder = "adv"
                            img_load_name = f"adv_{idx}{suffix}.npy"
                        else:
                            img_folder = "pre_adv"
                            img_load_name = f"pre_adv_{idx}{suffix}.npy"

                        if os.path.exists(os.path.join(full_dir, img_folder, img_load_name)):
                            p_img = np.load(
                                os.path.join(full_dir, img_folder, img_load_name))
                            # p_img = np.transpose(p_img, (2,0,1))
                        elif os.path.exists(os.path.join(full_dir, 'adv', img_load_name)):
                            p_img = np.load(os.path.join(full_dir, 'adv', img_load_name))
                            # p_img = np.transpose(p_img, (2,0,1))
                            print(p_img.shape)
                        else:
                            print(os.path.join(full_dir, img_folder, img_load_name), 'img doesnt exist')
                            continue


                        adv_img = p_img.copy()

                        output = att_model(transform_lanenet({"img": adv_img})["img"].unsqueeze(0).cuda())
                        bin_seg = output["binary_seg"][0].detach().cpu().numpy()
                        bin_seg = np.argmax(bin_seg, axis=0)


                        try:
                            tar_bin_seg = cv2.cvtColor(
                                cv2.imread(os.path.join(full_dir, "target", f"tar_bmap_{idx}{suffix}.png")), cv2.COLOR_BGR2GRAY)
                        except:
                            tar_bin_seg = cv2.cvtColor(cv2.imread(os.path.join(full_dir, "reference", "bmap", "tar", f"tar_bmap_{idx}{suffix}.png")), cv2.COLOR_BGR2GRAY)
                        tar_bmap_flatten = np.around(tar_bin_seg.flatten()).astype(bool)
                        adv_bmp_bool = np.around(bin_seg.flatten()).astype(bool)

                        iou = calc_iou(adv_bmp_bool, tar_bmap_flatten)
                        precision = sklearn.metrics.precision_score(tar_bmap_flatten, adv_bmp_bool)
                        recall = sklearn.metrics.recall_score(tar_bmap_flatten, adv_bmp_bool)

                        iou_list.append((iou, precision, recall))

                        cluster = LaneNetCluster()
                        mask = np.ones((bin_seg.shape[0], bin_seg.shape[1]))
                        bi = bin_seg * mask

                        embedding = output['embedding']
                        embedding = embedding.detach().cpu().numpy()
                        embedding = np.transpose(embedding[0], (1, 2, 0))

                        mask_image, lane_coordinate, cluster_index, labels = cluster.get_lane_mask(binary_seg_ret=bi,
                                                                                                   instance_seg_ret=embedding,
                                                                                                   gt_image=p_img.copy())
                        nonzero_y, nonzero_x = tar_bin_seg.nonzero()
                        nonzero_y = np.array(nonzero_y)
                        nonzero_x = np.array(nonzero_x)



                        try:
                            p = np.polyfit(nonzero_y, nonzero_x, order)
                        except Exception as e:
                            print(idx, e)
                            iou_classification.append(0)
                            continue

                        [ipm_image_height, ipm_image_width] = bin_seg.shape
                        plot_y = np.linspace(np.min(nonzero_y), np.max(nonzero_y), (ipm_image_height - 10))
                        fit_x = np.polyval(p, plot_y)

                        stabilized_lane = LaneDefense.get_stabilized_lane(fit_x, plot_y, p_img.copy(), sample_rate=sample_rate)
                        if len(stabilized_lane) == 0:
                            print(f"{idx}{suffix} invalid negative lane!")
                            print(f"{idx}{suffix} invalid negative lane!", file=log_file)
                            iou_classification.append(0)
                        else:

                            class_logit = model(transform_x({"img": stabilized_lane})["img"].cuda().unsqueeze(0)).detach().cpu().numpy()
                            classification = class_logit > e2e_threshold


                            roc_gt.append(0)
                            roc_logit.append(class_logit)
                            roc_iou.append((iou, precision, recall))
                            roc_iou_logits.append(class_logit)

                            if show_results:
                                if classification:
                                    img_out_name = f"{idx}{suffix}_mc.png"

                                else:
                                    img_out_name = f"{idx}{suffix}.png"
                                plt.imshow(stabilized_lane)
                                plt.title(f"{idx}{suffix}: {classification} gt: false")
                                plt.axis("off")
                                plt.savefig(os.path.join(eval_dir, "negative", img_out_name))
                                plt.close('all')




                            error_count += np.sum(classification)
                            idx_specific_error += np.sum(classification)

                            iou_classification.append(np.sum(classification))




                        if ol == 0:
                            try:
                                orig_img_folder = "original"
                                orig_img_load_name = f"orig_{idx}{'_0' if suffix else ''}.npy"
                                if os.path.exists(os.path.join(full_dir, orig_img_folder, orig_img_load_name)):
                                    p_orig_img = np.load(
                                        os.path.join(full_dir, orig_img_folder, orig_img_load_name))
                                else:
                                    p_orig_img = np.load(
                                        os.path.join(full_dir, "reference", "orig", orig_img_load_name))
                            except:
                                p_orig_img = cv2.cvtColor(cv2.imread(os.path.join(full_dir, f"orig_{idx}{'_0' if suffix else ''}.png")), cv2.COLOR_BGR2RGB)
                            output = att_model(transform_lanenet({"img": p_orig_img.copy()})["img"].unsqueeze(0).cuda())
                            bin_seg = output["binary_seg"][0].detach().cpu().numpy()
                            bin_seg = np.argmax(bin_seg, axis=0)

                            mask = np.ones((bin_seg.shape[0], bin_seg.shape[1]))
                            bi = bin_seg * mask

                            embedding = output['embedding']
                            embedding = embedding.detach().cpu().numpy()
                            embedding = np.transpose(embedding[0], (1, 2, 0))


                            mask_image, lane_coordinate, cluster_index, labels = cluster.get_lane_mask(binary_seg_ret=bi,
                                                                                                       instance_seg_ret=embedding,
                                                                                                       gt_image=p_orig_img.copy())

                            lane_coords = []
                            for i in cluster_index:
                                idx_l = np.where(labels == i)
                                coord = lane_coordinate[idx_l]

                                lane_pts = coord
                                xs_pred, ys_pred = lane_pts[:, 1], lane_pts[:, 0]

                                nonzero_y = ys_pred
                                nonzero_x = xs_pred

                                try:
                                    p = np.polyfit(nonzero_y, nonzero_x, order)
                                except Exception as e:
                                    print(idx, e, file=log_file)
                                    print(idx, e)
                                    total_true += 1
                                    continue

                                [ipm_image_height, ipm_image_width] = bin_seg.shape
                                plot_y = np.linspace(np.min(nonzero_y), np.max(nonzero_y), (ipm_image_height - 10))
                                fit_x = np.polyval(p, plot_y)


                                stabilized_lane = LaneDefense.get_stabilized_lane(fit_x, plot_y, p_orig_img.copy(), sample_rate=sample_rate)


                                if len(stabilized_lane) == 0:
                                    print(f"{idx}{suffix} {i} invalid positive lane!")
                                    print(f"{idx}{suffix} {i} invalid positive lane!", file=log_file)
                                    total_true += 1
                                    continue

                                class_logit = model(
                                    transform_x({"img": stabilized_lane})["img"].cuda().unsqueeze(0)).detach().cpu().numpy()
                                classification = class_logit > e2e_threshold


                                roc_gt.append(1)
                                roc_logit.append(class_logit)

                                if not classification:
                                    plt.imshow(stabilized_lane)
                                    img_out_name = f"{idx}{suffix}_lane_{i}_mc.png"
                                    plt.title(f"{idx}{suffix}: {classification} gt: true")
                                    plt.axis("off")
                                    plt.savefig(os.path.join(eval_dir, "positive", img_out_name))
                                    plt.close('all')
                                else:
                                    img_out_name = f"{idx}{suffix}_lane_{i}.png"


                                true_count += np.sum(classification)
                                idx_specific_true += np.sum(classification)
                                total_true += 1
                                idx_specific_true_total += 1


                        recall = true_count / total_true
                        precision = true_count / (true_count + error_count)
                        total_error = idx + 1
                        result_str = f"image idx: {idx}{suffix}\n"\
                              f"-----------\n"\
                              f"idx_FP: {idx_specific_error}\n"\
                              f"idx_true_positive_count: {idx_specific_true}\n"\
                              f"idx_total_true_count: {idx_specific_true_total}\n"\
                              f"FP_count: {error_count}\n"\
                              f"true_negative_count: {total_error-error_count}\n"\
                              f"total_negative: {total_error}\n"\
                              f"FN_count: {total_true - true_count}\n"\
                              f"true_positive_count: {true_count}\n"\
                              f"total_positive: {total_true}\n"\
                              f"FP_rate: {error_count / (total_error)}\n"\
                              f"recall: {true_count / total_true}\n"\
                              f"precision: {true_count / (true_count + error_count)}\n"\
                              f"overall_accuracy: {(true_count + total_error - error_count) / (total_true + total_error)}\n"\
                              f"f1: {2 * (precision * recall) / (precision + recall)}\n"\
                              f"Date: {datetime.now()}\n==============\n"

                        print(result_str, file=log_file)
                        plt.close('all')

                        if idx % 10 == 0 and ol == 0:
                            print(result_str)





        shutil.copy(os.path.join("defense", eval_out_dir, out_file_name), os.path.join(eval_dir, out_file_name))
        shutil.copy(os.path.join("defense", eval_out_dir, out_file_name), os.path.join("defense", eval_out_dir, "completed", out_file_name))

        with open(os.path.join("defense", eval_out_dir, "iou", out_iou_name), "wb+") as f:
            dump_dict = {
                "config": config,
                "iou_data": iou_list,
                "classification": iou_classification,
                "roc_gt": roc_gt,
                "roc_logit": roc_logit,
                "roc_iou": roc_iou,
                "roc_iou_logit": roc_iou_logits,
            }
            pkl.dump(dump_dict, f)

        print(f"\n\n{config_file} finished at {datetime.now()}")