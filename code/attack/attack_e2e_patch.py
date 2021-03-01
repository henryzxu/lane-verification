import json
import time
from datetime import datetime

import sklearn.metrics
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from verifier.defense import LaneDefense
from verifier.defense_dataset import DefenseDataset
from torch.utils.data import DataLoader

from verifier.defense_nn_models import DefenseMLP, DefenseLinear, DefenseMaxPool
from utils.tensorboard import TensorBoard
from utils.transforms import *
import os


import torchvision.transforms.functional as F
import torch.nn.functional as F2
import matplotlib.pyplot as plt

from attack.lanenet_attack import lanenet_att_model, get_targets, lanenet_loss, save_image_from_raw_input,\
    calc_iou, build_data_loader

from attack.lanenet_patch_helpers import *

def save_image_from_raw_bitmap(raw_binary_seg, fname, take_argmax=True, mask=None, target=None):
    bin_seg_prob = raw_binary_seg.detach().cpu().numpy()
    if take_argmax:
        bin_seg_pred = np.argmax(bin_seg_prob, axis=0)
    else:
        bin_seg_pred = bin_seg_prob

    bin_seg_img = np.zeros((288, 512, 3), dtype=np.uint8)
    bin_seg_img[bin_seg_pred==1, :] = (255, 255, 255)
    # bin_seg_img = cv2.cvtColor(bin_seg_img, cv2.COLOR_BGR2RGB)



    if target is not None:
        target = target.detach().cpu().numpy()
        sparse_filter = np.random.randn(target.shape[0], target.shape[1]) > 0.5
        target = target * sparse_filter
        bin_seg_img[np.nonzero(target)] = (0, 255, 0)

    if mask is not None:
        mask_nonzero = np.transpose(np.nonzero(mask))
        color = (255, 0, 0)
        thickness = 2

        bin_seg_img = cv2.rectangle(bin_seg_img, tuple([int(x) for x in mask_nonzero[0][1:]][::-1]), tuple([int(x) for x in mask_nonzero[-1][1:]][::-1]), color, thickness=thickness)
    save_image_from_raw_input(bin_seg_img, fname, permute=False)


config_file = "experiments/exp0/verifier_model/exp115a2000_patch/exp115a2000_advtrain_2020-08-21__21-31-09_.config"


device = torch.device("cuda:1")


if os.path.exists(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
        mean = config["mean"]
        std = config["std"]

        kernel_size = config["kernel_size"]
        exp_num = config["exp_num"]
        hidden_layer1 = config["hidden_layer1"]
        hidden_layer2 = config["hidden_layer2"]
        model_type = config["model_type"]
        additional_modifiers = config["additional_modifiers"]
        noise_sd = config["noise_sd"]
        exp = config["exp"]
        train_dataset = config["train_dataset"]
        val_dataset = config['val_dataset']
        dataset_ext = config['dataset_ext']
        pretrained_epoch = config.get("pretrained_epoch", 90)
        adv_train = config.get("adv_train", False)
        adv_exp = config.get("adv_exp", None)
        adv_eps = config.get('adv_eps')
        adv_iter = config.get('adv_iter', 3)
        e2e_expnum = config.get('e2e_expnum', -1)
        sample_rate = config.get("sample_rate", 0.5)
        adv_floss_gamma = config.get("adv_floss_gamma", 2)
        adv_floss_alpha = config.get("adv_floss_alpha", 0.75)
        adv_bce_loss = config.get("adv_bce_loss", 2)
        one_way = config.get("one_way", False)
        full_exp = config.get("full_exp", None)
        patch_height = config.get("patch_height", None)
        min_patch = config.get("min_patch", None)
        patch_shape = config.get("patch_shape_cv2", None)
else:
    raise AssertionError("config file doesn't exist!")

if full_exp:
    exp = full_exp

config["lanenet_eps"] = lanenet_eps = 8/255
config["lanenet_nb_iter"] = lanenet_nb_iter = 100
config["lanenet_eps_iter"] = lanenet_eps_iter = 5

if adv_eps is None:
    config["e2e_adv_eps"] = adv_eps = 8/255

config["e2eexp"] = config["e2e_expnum"] = e2eexp = 303
config["e2e_resume_epoch"] = resume_epoch = 8
config["e2e_classifier_adv_iter"] = adv_iter = 10
config["e2e_outerloop"] = outer_loop_count = 4
config["e2e_starting_batch_idx"] = starting_batch_idx = 0
config["e2e_finish_batch_idx"] = finish_batch_idx = -1
config["e2e_lanenet_dset"] = lanenet_dset = "val"
config["e2e_supplied_config"] = config_file
config["e2e_inner_optimization_loop_count"] = inner_optimization_loop_count = 1
config["e2e_bce_loss"] = e2e_bce_loss = 1
config["e2e_gen_notes"] = "attack_images_v2 PATCH ATTACK no resize thresh test fixed"
config["full_exp"] = exp + f"e{e2eexp}"
config["patch_mask_length"] = mask_length = 100
config["patch_lr"] = patch_lr = 1e2
config["patch_iterations"] = patch_iterations = 100
config["fixed_size"] = fixed_patch = True



if "patch" not in config_file:
    resize_shape = tuple(config["resize_shape"])

if __name__=='__main__':
    transform_x = Compose( ToTensor())



    if "patch" in config_file:
        if model_type == "conv":
            model =  DefenseMLP(kernel_size=kernel_size, input_shape=patch_shape).to(device)
        elif model_type =="conv_maxpool":
            model =  DefenseMaxPool(kernel_size=kernel_size, input_shape=patch_shape).to(device)
        else:
            model = DefenseLinear(3*np.prod(patch_shape), hidden_layer1, hidden_layer2).to(device)
    else:
        if model_type == "conv":
            model = DefenseMLP(kernel_size=kernel_size, input_shape=resize_shape).to(device)
        elif model_type == "conv_maxpool":
            model = DefenseMaxPool(kernel_size=kernel_size, input_shape=resize_shape).to(device)
        else:
            model = DefenseLinear(3 * np.prod(resize_shape), hidden_layer1, hidden_layer2).to(device)




    class NormalizeModel(nn.Module):
        def __init__(
                self,
                base_model
        ):
            super(NormalizeModel, self).__init__()
            self.base = base_model


        def forward(self, img):

            img = torch.stack([F.normalize(i, mean, std) for i in img])
            if "patch" not in config_file:
                img = nn.AdaptiveAvgPool2d(resize_shape[::-1])(img)

            x = self.base(img)  # x is a dict
            return x

    tensorboard = SummaryWriter(log_dir="defense/attack/logs/" + exp)


    out_dir_perturb = os.path.join("defense", "attack", "e2e", exp + f"e{e2eexp}")
    config["e2e_out_dir"] = out_dir_perturb


    if adv_exp:
        out_dir = os.path.join("defense", "attack", "saved_models", exp)
        save_name = os.path.join(out_dir, str(resume_epoch) + '_advtrain.pth')
        config["e2e_epoch"] = resume_epoch
    else:
        save_name = os.path.join("defense", "saved_models", exp, str(pretrained_epoch) + '.pth')
        resume_epoch = 1
        config["e2e_epoch"] = pretrained_epoch
    config["e2e_save_name"] = save_name

    print("loading {}".format(save_name))

    save_dict = torch.load(save_name)

    model.load_state_dict(save_dict['net'])


    att_model = NormalizeModel(model)

    att_model.eval()

    def criterion(predictions, targets, gamma=adv_floss_gamma, alpha=adv_floss_alpha):
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1).cuda(device=device) * e2e_bce_loss)(predictions, targets)
        alpha_tensor = (1-alpha) + targets * (2 * alpha - 1)
        p_t = torch.exp(-bce_loss)
        focal_loss = alpha_tensor * torch.pow(1-p_t, gamma) * bce_loss
        return bce_loss

    optimizer = torch.optim.Adam(att_model.parameters(), weight_decay=1e-3, lr=1e-3)


    os.makedirs(out_dir_perturb, exist_ok=True)
    os.makedirs(os.path.join(out_dir_perturb, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_perturb, "iou_data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_perturb, "target"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_perturb, "adv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_perturb, "pre_adv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir_perturb, "original"), exist_ok=True)



    count = 0

    lanenet_loader = build_data_loader(lanenet_dset, batch_size=1)

    out_time = time.strftime("%Y-%m-%d__%H-%M-%S")

    with open(os.path.join("defense/configs", f"{exp}e{e2eexp}_e2e_{out_time}.config"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(out_dir_perturb, f"{exp}e{e2eexp}_e2e_{out_time}.config"), "w") as f:
        json.dump(config, f)


    with open(os.path.join(out_dir_perturb, "logs", f"{exp}e{e2eexp}_{out_time}.log"), "a+") as log_file:
        print("loaded {}".format(save_name), file=log_file)
        print(json.dumps(config, indent=4, sort_keys=True), file=log_file)
        print(json.dumps(config, indent=4, sort_keys=True))

        for batch_idx, sample in enumerate(tqdm(lanenet_loader)):
            i = 0
            outer_count = 0
            if finish_batch_idx > 0 and batch_idx >= finish_batch_idx:
                exit()
            outer_images = sample['img'].cuda(device=device)

            if batch_idx < starting_batch_idx:
                count += outer_images.shape[0]
                continue

            images = outer_images.clone().detach()
            num_img = images.shape[0]
            original_images = images.clone().detach()

            counter = 0

            while counter <= 10:
                try:
                    guides = get_targets(num_img)
                    applied_patch, mask, x_location, y_location = mask_generation(guides, mask_length, fixed=fixed_patch)
                except AssertionError:
                    counter += 1
                else:
                    break
            assert counter <= 10

            images, applied_patch = patch_attack(images, applied_patch, mask, guides, lanenet_loss,
                                                 lanenet_att_model, device, lr=patch_lr, max_iteration=patch_iterations)


            adv_bmaps = lanenet_att_model(images).detach().cpu()


            adv = images[0].permute(1, 2, 0)
            orig = original_images[0].permute(1, 2, 0)
            applied_patch_permute = applied_patch[0].permute(1,2,0)
            mask_permute = np.transpose(mask, (1,2,0))

            adv_bmap = adv_bmaps[0]
            tar_bmap = guides[0]

            adv_bmp_bool = np.around(np.argmax(adv_bmap.detach().cpu().numpy(), axis=0)).flatten().astype(bool)
            tar_bmap_flatten = np.around(tar_bmap.detach().cpu().numpy().flatten()).astype(bool)

            save_image_from_raw_bitmap(adv_bmap, "{}/reference/bmap/patch/patch_bmap_{}_{}".format(out_dir_perturb, count + i, outer_count), mask=mask, target=tar_bmap)
            save_image_from_raw_bitmap(tar_bmap, "{}/reference/bmap/tar/tar_bmap_{}_{}".format(out_dir_perturb, count + i, outer_count), take_argmax=False)
            save_image_from_raw_input(adv.cpu(), "{}/pre_adv/pre_adv_{}_{}".format(out_dir_perturb, count + i, outer_count), numpy_save=True)


            orig_bmaps = lanenet_att_model(original_images).detach().cpu()
            orig_bmap = orig_bmaps[0]
            orig_bmp_bool = np.around(np.argmax(orig_bmap.detach().cpu().numpy(), axis=0)).flatten().astype(bool)
            save_image_from_raw_bitmap(orig_bmap, "{}/reference/bmap/orig/original_bmp_{}_{}".format(out_dir_perturb, count + i, outer_count), mask=mask, target=tar_bmap)

            iou = calc_iou(adv_bmp_bool, tar_bmap_flatten)
            precision = sklearn.metrics.precision_score(tar_bmap_flatten, adv_bmp_bool)
            recall = sklearn.metrics.recall_score(tar_bmap_flatten, adv_bmp_bool)





            bin_seg = tar_bmap.detach().cpu().numpy()
            bin_nonzero_y, bin_nonzero_x = bin_seg.nonzero()
            nonzero_y = np.array(bin_nonzero_y)
            nonzero_x = np.array(bin_nonzero_x)

            try:
                p = np.polyfit(nonzero_y, nonzero_x, 3)
            except Exception as e:
                print(count, e)
                continue

            [ipm_image_height, ipm_image_width] = bin_seg.shape

            plot_y = np.linspace(np.min(nonzero_y), np.max(nonzero_y), (ipm_image_height - 10))
            fit_x = np.polyval(p, plot_y)



            stabilized_lane = LaneDefense.get_stabilized_lane(fit_x, plot_y, adv, only_index=True, sample_rate=sample_rate)

            stacked_lines = []
            adv_starting_point = []
            stable_mask = []
            stable_applied_patch = []
            for extract_idx in stabilized_lane:
                row, col = extract_idx
                stacked_lines.append(orig[row,col])
                adv_starting_point.append(adv[row, col])
                stable_mask.append(mask_permute[row, col])
                stable_applied_patch.append(applied_patch_permute[row, col])

            extracted_lane = torch.stack(stacked_lines)
            extracted_adv_lane = torch.stack(adv_starting_point)
            mask_lane = torch.from_numpy(np.array(stable_mask) > 0).type(torch.FloatTensor).cuda(device=device)
            applied_patch_lane = torch.stack(stable_applied_patch)



            stabilized_orig_tensor = extracted_lane.permute(2, 0, 1)
            stabilized_orig_tensor = stabilized_orig_tensor.cuda(device=device).unsqueeze(0)
            original_stable_tensor = stabilized_orig_tensor.clone().detach()

            stabilized_tensor = extracted_adv_lane.permute(2, 0, 1)
            stabilized_tensor = stabilized_tensor.cuda(device=device).unsqueeze(0)


            mask_tensor = mask_lane.permute(2, 0, 1)

            applied_patch_tensor = applied_patch_lane.permute(2, 0, 1)


            if 'patch' in config_file:
                stabilized_tensor, lane_applied_patch = patch_attack_lane(stabilized_tensor, applied_patch_tensor.detach().cpu().numpy(),
                                                                 mask_tensor.detach().cpu().numpy(), patch_shape,
                                                                 torch.ones(1,1).to(device), criterion, att_model, device,
                                                                          lr=patch_lr, max_iteration=patch_iterations)
            else:
                stabilized_tensor, lane_applied_patch = patch_attack_orig_classifier(stabilized_tensor, applied_patch_tensor.detach().cpu().numpy(),
                                                                 mask_tensor.detach().cpu().numpy(), patch_shape,
                                                                 torch.ones(1,1).to(device), criterion, att_model, device,
                                                                          lr=patch_lr, max_iteration=patch_iterations)

            print(np.count_nonzero((original_stable_tensor - stabilized_tensor).detach().cpu().numpy()), file=log_file)

            stable_result = stabilized_tensor[0].permute(1, 2, 0)
            for extract_idx in range(len(stabilized_lane)):
                row, col = stabilized_lane[extract_idx]
                adv[row, col] = stable_result[extract_idx]


            adv_test = []
            for extract_idx in stabilized_lane:
                row, col = extract_idx
                adv_test.append(images[0].permute(1, 2, 0)[row, col])

            extracted_adv_lane_test = torch.stack(adv_test)


            stabilized_tensor_test = extracted_adv_lane_test.permute(2, 0, 1)
            stabilized_tensor_test = stabilized_tensor_test.cuda(device=device).unsqueeze(0)


            adv_bmaps = lanenet_att_model(images).detach().cpu()

            i = 0
            adv = images[i].cpu()
            orig = original_images[i].cpu()

            adv_bmap = adv_bmaps[i]
            tar_bmap = guides[i]


            adv_permute = adv.permute(1, 2, 0)
            orig_permute = orig.permute(1, 2, 0)
            adv_bmap_numpy = adv_bmap.detach().cpu().numpy()
            tar_bmap_numpy = tar_bmap.detach().cpu().numpy()

            adv_bmap_argmax = np.around(np.argmax(adv_bmap_numpy, axis=0))
            adv_bmp_bool = adv_bmap_argmax.flatten().astype(bool)
            tar_bmap_flatten = np.around(tar_bmap_numpy.flatten()).astype(bool)

            iou = calc_iou(adv_bmp_bool, tar_bmap_flatten)
            precision = sklearn.metrics.precision_score(tar_bmap_flatten, adv_bmp_bool)
            recall = sklearn.metrics.recall_score(tar_bmap_flatten, adv_bmp_bool)

            new_iou = (iou, precision, recall)





            save_image_from_raw_input(adv, "{}/adv/adv_{}_{}".format(out_dir_perturb, count + i, outer_count), numpy_save=True)

            if outer_count == 0:
                save_image_from_raw_input(orig, "{}/reference/orig/orig_{}_{}".format(out_dir_perturb, count + i, outer_count), numpy_save=True)


            log_file.flush()
            plt.close(fig="all")


            print(f"finished {count}-{count + num_img - 1} at {datetime.now()}")
            print(f"finished {count}-{count + num_img - 1} at {datetime.now()}", file=log_file)
            count += num_img

        exit()



        print(f"finished e2e training at {datetime.now()}", file=log_file)











# Epoch 0
# -----------
# Val loss: 0.0000 Val accuracy: 0.5382 Val FP: 0.400436 Val FN: 0.6377459749552773
# ==============


# Epoch 0
# -----------
# Val loss: 0.0000 Val accuracy: 0.9734 Val FP: 0.007791 Val FN: 0.08050089445438283
# ==============