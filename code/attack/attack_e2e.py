import json
import time
from datetime import datetime

import sklearn.metrics
import torch
# from advertorch.context import ctx_noparamgrad_and_eval
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

# from advertorch.attacks import LinfPGDAttack

import torchvision.transforms.functional as F
import torch.nn.functional as F2
import matplotlib.pyplot as plt
# from skimage.transform import resize

from attack.lanenet_attack import lanenet_att_model, get_targets, lanenet_loss, save_image_from_raw_input, \
    save_image_from_raw_bitmap, calc_iou, build_data_loader, device

config_file = "experiments/exp0/verifier_model/exp115a3000_l_infinity/exp115a3000_advtrain_2020-12-29__01-46-45_R.config"

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
else:
    raise AssertionError("config file doesn't exist!")

if full_exp:
    exp = full_exp

config["lanenet_eps"] = lanenet_eps = 8/255
config["lanenet_nb_iter"] = lanenet_nb_iter = 100
config["lanenet_eps_iter"] = lanenet_eps_iter = lanenet_eps * 2/lanenet_nb_iter

if adv_eps is None:
    config["e2e_adv_eps"] = adv_eps = 8/255

config["e2eexp"] = config["e2e_expnum"] = e2eexp = 101
config["e2e_resume_epoch"] = resume_epoch = 46
config["e2e_classifier_adv_iter"] = adv_iter = 40
config["e2e_outerloop"] = outer_loop_count = 1
config["e2e_starting_batch_idx"] = starting_batch_idx = 0
config["e2e_finish_batch_idx"] = finish_batch_idx = -1
config["e2e_lanenet_dset"] = lanenet_dset = "val"
config["e2e_supplied_config"] = config_file
config["e2e_inner_optimization_loop_count"] = inner_optimization_loop_count = 1
config["e2e_bce_loss"] = e2e_bce_loss = 1
config["e2e_gen_notes"] = "verification val 30000"
config["full_exp"] = exp + f"e{e2eexp}"

if __name__=='__main__':
    transform_x = Compose(ToTensor())

    if model_type == "conv":
        model =  DefenseMLP(kernel_size=kernel_size, input_shape=resize_shape).to(device)
    elif model_type =="conv_maxpool":
        model =  DefenseMaxPool(kernel_size=kernel_size, input_shape=resize_shape).to(device)
    else:
        model = DefenseLinear(3*np.prod(resize_shape), hidden_layer1, hidden_layer2).to(device)

    class NormalizeModel(nn.Module):
        def __init__(
                self,
                base_model
        ):
            super(NormalizeModel, self).__init__()
            self.base = base_model
            self.resize = nn.AdaptiveAvgPool2d(resize_shape[::-1])

        def forward(self, img):
            img = torch.stack([F.normalize(i, mean, std) for i in img])
            img = self.resize(img)
            # img = F2.interpolate(img, size=resize_shape[::-1], mode='area')
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
        if model_type == 'baseline':
            return torch.nn.MSELoss()(predictions, targets)
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1).cuda(device=device) * e2e_bce_loss)(predictions, targets)
        alpha_tensor = (1-alpha) + targets * (2 * alpha - 1)
        p_t = torch.exp(-bce_loss)
        focal_loss = alpha_tensor * torch.pow(1-p_t, gamma) * bce_loss
        # return focal_loss.mean()
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

    lanenet_loader = build_data_loader(lanenet_dset, batch_size=4)

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
            if finish_batch_idx > 0 and batch_idx >= finish_batch_idx:
                exit()
            outer_images = sample['img'].cuda(device=device)

            if batch_idx < starting_batch_idx:
                count += outer_images.shape[0]
                continue
            for outer_count in range(outer_loop_count):
                images = outer_images.clone().detach()
                num_img = images.shape[0]
                original_images = images.clone().detach()
                guides = get_targets(num_img)

                eta = torch.FloatTensor(images.shape).uniform_(-lanenet_eps, lanenet_eps).cuda(device=device)
                images = torch.clamp(original_images + eta, min=0, max=1).detach()

                iou_data = []

                for inner_op in range(inner_optimization_loop_count):
                    for lanenet_adv_iter in range(lanenet_nb_iter):
                        images.requires_grad = True
                        outputs = lanenet_att_model(images)

                        lanenet_att_model.zero_grad()
                        cost = -lanenet_loss(outputs, guides).to(device)
                        cost.backward()

                        adv_images = images + lanenet_eps_iter * images.grad.sign()
                        eta = torch.clamp(adv_images - original_images, min=-lanenet_eps, max=lanenet_eps)
                        images = torch.clamp(original_images + eta, min=0, max=1).detach()

                        adv_bmaps = lanenet_att_model(images).detach().cpu()

                        iou_iter = []

                        if lanenet_adv_iter == lanenet_nb_iter - 1:
                            pre_classifier_adv = images.clone().detach()

                        for i in range(num_img):
                            adv = images[i].permute(1, 2, 0)
                            orig = original_images[i].permute(1, 2, 0)
                            adv_bmap = adv_bmaps[i].detach().cpu().numpy()
                            tar_bmap = guides[i].detach().cpu().numpy()

                            adv_bmap = np.around(np.argmax(adv_bmap, axis=0))
                            adv_bmp_bool=adv_bmap.flatten().astype(bool)
                            tar_bmap_flatten = np.around(tar_bmap.flatten()).astype(bool)

                            iou = calc_iou(adv_bmp_bool, tar_bmap_flatten)
                            precision = sklearn.metrics.precision_score(tar_bmap_flatten, adv_bmp_bool)
                            recall = sklearn.metrics.recall_score(tar_bmap_flatten, adv_bmp_bool)

                            iou_iter.append((iou, precision, recall))

                            if lanenet_adv_iter == lanenet_nb_iter - 1:
                                # print("starting classifier attack of batch", batch_idx, "image", i)
                                # pre_classifier_adv = images.clone().detach()

                                bin_seg = adv_bmap
                                nonzero_y = np.array(bin_seg.nonzero()[0])
                                nonzero_x = np.array(bin_seg.nonzero()[1])

                                try:
                                    p = np.polyfit(nonzero_y, nonzero_x, 3)
                                except Exception as e:
                                    print(count, lanenet_adv_iter, i, e)
                                    continue

                                [ipm_image_height, ipm_image_width] = bin_seg.shape
                                plot_y = np.linspace(np.min(nonzero_y), np.max(nonzero_y), (ipm_image_height - 10))
                                fit_x = np.polyval(p, plot_y)


                                stabilized_lane = LaneDefense.get_stabilized_lane(fit_x, plot_y, adv, only_index=True, sample_rate=sample_rate)

                                stacked_lines = []
                                adv_starting_point = []
                                for extract_idx in stabilized_lane:
                                    row, col = extract_idx
                                    stacked_lines.append(orig[row,col])
                                    adv_starting_point.append(adv[row, col])
                                extracted_lane = torch.stack(stacked_lines)
                                extracted_adv_lane = torch.stack(adv_starting_point)

                                stabilized_orig_tensor = extracted_lane.permute(2, 0, 1)
                                stabilized_orig_tensor = stabilized_orig_tensor.cuda(device=device).unsqueeze(0)
                                original_stable_tensor = stabilized_orig_tensor.clone().detach()

                                stabilized_tensor = extracted_adv_lane.permute(2, 0, 1)
                                stabilized_tensor = stabilized_tensor.cuda(device=device).unsqueeze(0)

                                for adv_classifier_iter in range(adv_iter):
                                    stabilized_tensor.requires_grad = True
                                    outputs_stabilized = att_model(stabilized_tensor)
                                    print(count, adv_iter, i, adv_classifier_iter, outputs_stabilized > 0, file=log_file)

                                    att_model.zero_grad()
                                    cost_stabilized = -criterion(outputs_stabilized, torch.ones(1,1).to(device))
                                    cost_stabilized.backward()

                                    adv_stable = stabilized_tensor + adv_eps * 2/adv_iter * stabilized_tensor.grad.sign()
                                    eta_stable = torch.clamp(adv_stable - original_stable_tensor, min=-adv_eps, max=adv_eps)
                                    stabilized_tensor = torch.clamp(original_stable_tensor + eta_stable, min=0, max=1).detach()

                                print(np.count_nonzero((original_stable_tensor - stabilized_tensor).detach().cpu().numpy()), file=log_file)

                                stable_result = stabilized_tensor[0].permute(1, 2, 0)
                                for extract_idx in range(len(stabilized_lane)):
                                    row, col = stabilized_lane[extract_idx]
                                    adv[row, col] = stable_result[extract_idx]

                                adv_test = []
                                for extract_idx in stabilized_lane:
                                    row, col = extract_idx
                                    adv_test.append(images[i].permute(1, 2, 0)[row, col])

                                extracted_adv_lane_test = torch.stack(adv_test)

                                stabilized_tensor_test = extracted_adv_lane_test.permute(2, 0, 1)
                                stabilized_tensor_test = stabilized_tensor_test.cuda(device=device).unsqueeze(0)
                                print("remap", count, i, outer_count, att_model(stabilized_tensor_test) > 0, file=log_file)

                                print("orig differences", outer_count,  np.count_nonzero((original_images[i] - adv.permute(2,0,1)).detach().cpu().numpy()), file=log_file)
                                print("view check",outer_count,  np.count_nonzero((images[i] - adv.permute(2,0,1)).detach().cpu().numpy()), file=log_file)
                                print("classifier differences", outer_count, np.count_nonzero((pre_classifier_adv[i] - adv.permute(2,0,1)).detach().cpu().numpy()), file=log_file)

                        iou_data.append(iou_iter)

                iou_data = np.array(iou_data)

                f, axs = plt.subplots(num_img, 1, sharex=True)
                ravelled_ax = axs.ravel()
                for ax_idx in range(len(ravelled_ax)):
                    ax = ravelled_ax[ax_idx]
                    ax.plot(range(lanenet_nb_iter * inner_optimization_loop_count), iou_data[:,ax_idx, 0], label="iou")
                    ax.plot(range(lanenet_nb_iter * inner_optimization_loop_count), iou_data[:,ax_idx, 1], label="precision")
                    ax.plot(range(lanenet_nb_iter * inner_optimization_loop_count), iou_data[:,ax_idx, 2], label="recall")
                    ax.set_title(ax_idx)

                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow = True, ncol = 2)

                f.suptitle(f"Adv Metrics vs Adversarial Iterations Images {count}-{count+num_img-1} Loop {outer_count}")


                plt.savefig(f"{out_dir_perturb}/iou_data/iou_plot_{count}-{count+num_img-1}_{outer_count}.png")



                adv_bmaps = lanenet_att_model(images).detach().cpu()
                pre_classifier_adv_bmaps = lanenet_att_model(pre_classifier_adv).detach().cpu()

                for i in range(num_img):
                    adv = images[i].cpu()
                    orig = original_images[i].cpu()

                    adv_bmap = adv_bmaps[i]
                    tar_bmap = guides[i]

                    pre_classier_img = pre_classifier_adv[i].cpu()
                    pre_classifier_adv_bmap = pre_classifier_adv_bmaps[i]

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

                    original_iou = iou_data[-1][i]
                    new_iou = (iou, precision, recall)

                    print("preclassifer", count, i, outer_count, original_iou, file=log_file)
                    print("postclassifer", count, i, outer_count,  new_iou, file=log_file)


                    save_image_from_raw_input(adv, "{}/adv/adv_{}_{}".format(out_dir_perturb, count + i, outer_count), numpy_save=True)
                    save_image_from_raw_input(pre_classier_img, "{}/pre_adv/pre_adv_{}_{}".format(out_dir_perturb, count + i, outer_count), numpy_save=True)

                    save_image_from_raw_bitmap(tar_bmap, "{}/target/tar_bmap_{}_{}".format(out_dir_perturb, count + i, outer_count), take_argmax=False)
                    if outer_count == 0:
                        save_image_from_raw_input(orig, "{}/original/orig_{}_{}".format(out_dir_perturb, count + i, outer_count), numpy_save=True)


                    log_file.flush()
                    plt.close(fig="all")


            print(f"finished {count}-{count + num_img - 1} at {datetime.now()}")
            print(f"finished {count}-{count + num_img - 1} at {datetime.now()}", file=log_file)
            count += num_img



        print(f"finished e2e training at {datetime.now()}", file=log_file)









