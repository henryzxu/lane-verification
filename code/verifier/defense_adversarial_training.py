import json
import time
from datetime import datetime

import torch
from advertorch.context import ctx_noparamgrad_and_eval
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from verifier.defense_dataset import DefenseDataset
from torch.utils.data import DataLoader

from verifier.defense_nn_models import DefenseMLP, DefenseLinear, DefenseMaxPool
from utils.transforms import *
import os

from advertorch.attacks import LinfPGDAttack

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

os.chdir("../../")

config_file = "defense\configs\exp115_nntrain_2020-08-09__00-14-05.config"
do_train = True



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
        pretrained_epoch = config.get("pretrained_epoch", 10)
        adv_train = config.get("adv_train", False)
        adv_exp = config.get("adv_exp", None)
        adv_eps = config.get('adv_eps', 5/255)
        adv_iter = config.get('adv_iter', 3)
        e2e_expnum = config.get('e2e_expnum', -1)
        sample_rate = config.get("sample_rate", 0.5)
        adv_floss_gamma = config.get("adv_floss_gamma", 1)
        adv_floss_alpha = config.get("adv_floss_alpha", 0.01)
        adv_bce_loss = config.get("adv_bce_loss", 1)
        one_way = config.get("one_way", False)
        interpolation = config.get("interpolation", "INTER_AREA")
        interpolation_cv2 = config.get("interpolation_cv2", cv2.INTER_AREA)
else:
    raise AssertionError("config file doesn't exist!")

config["adv_resume_epoch"] = resume_epoch = 4
config["adv_supplied_config"] = config_file

if not adv_exp:
    config["adv_exp"] = advexp = 4000
    config["adv_eps"] = eps = 8 / 255
    config["adv_iter"] = nb_iter = 40
    config["adv_floss_gamma"] = adv_floss_gamma
    config["adv_floss_alpha"] = adv_floss_alpha
    config["adv_bce_loss"] = adv_bce_loss
    config["one_way"] = one_way = True
    config["adv_train"] = True
    config["adv_additional_notes"] = "removed detach from orig_img"
else:
    advexp = adv_exp
    eps = adv_eps
    nb_iter = adv_iter


if __name__=='__main__':

    initialization_out = []
    transform_x = Compose(Resize(resize_shape, interpolation=interpolation_cv2), ToTensor())
    train_loader = DataLoader(DefenseDataset("defense", train_dataset, transforms=transform_x, ext=dataset_ext), pin_memory=True, batch_size=256, shuffle=True, collate_fn=DefenseDataset.collate)

    val_loader = DataLoader(DefenseDataset("defense", val_dataset, transforms=transform_x, ext=dataset_ext), pin_memory=True, batch_size=256, shuffle=True, collate_fn=DefenseDataset.collate)


    device = torch.device("cuda:0")


    if model_type == "conv":
        model = DefenseMLP(kernel_size=kernel_size, input_shape=resize_shape).to(device)
    elif model_type =="conv_maxpool":
        model = DefenseMaxPool(kernel_size=kernel_size, input_shape=resize_shape).to(device)
    else:
        model = DefenseLinear(3*np.prod(resize_shape), hidden_layer1, hidden_layer2).to(device)



    class NormalizeModel(nn.Module):
        def __init__(
                self,
                base_model
        ):
            super(NormalizeModel, self).__init__()
            self.base = base_model

        def forward(self, img):
            img = torch.stack([F.normalize(i, mean, std) for i in img])
            x = self.base(img)  # x is a dict
            return x




    orig_exp = exp


    exp = orig_exp + f"a{advexp}"
    config["full_exp"] = exp


    tensorboard = SummaryWriter(log_dir="defense/attack/logs/" + exp)

    out_dir = os.path.join("defense", "attack", "saved_models", exp)



    save_name = os.path.join(out_dir, str(resume_epoch) + '_advtrain.pth')

    config["adv_resume"] = resume = False
    if not os.path.exists(save_name) and (not os.path.exists(out_dir) or len(os.listdir(out_dir)) == 0):
        save_name = os.path.join("defense", "saved_models", orig_exp, str(pretrained_epoch) + '.pth')
        resume_epoch = 0
    elif not os.path.exists(save_name) and os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(save_name)
        raise AssertionError("advtrain output not empty!")
    else:
        config["adv_resume"] = resume = True

    config["advtrain_save_name"] = save_name

    initialization_out.append("loading {}".format(save_name))
    print(initialization_out[0])

    save_dict = torch.load(save_name)

    model.load_state_dict(save_dict['net'])


    att_model = NormalizeModel(model)


    def criterion(predictions, targets, gamma=adv_floss_gamma, alpha=adv_floss_alpha):
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1).cuda(device=device) * adv_bce_loss)(predictions, targets)
        alpha_tensor = (1-alpha) + targets * (2 * alpha - 1)
        p_t = torch.exp(-bce_loss)
        focal_loss = alpha_tensor * torch.pow(1-p_t, gamma) * bce_loss
        return focal_loss.mean()

    adversary = LinfPGDAttack(
        att_model, loss_fn=criterion, eps=eps,
        nb_iter=nb_iter, eps_iter=eps * 2/nb_iter, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)


    optimizer = torch.optim.Adam(att_model.parameters(), weight_decay=1e-3, lr=1e-3)


    os.makedirs(out_dir, exist_ok=True)

    initialization_out.append(f"out directory: {out_dir}")
    print(initialization_out[1])

    out_file_time = time.strftime("%Y-%m-%d__%H-%M-%S")
    out_file_name = f"advtrain_{'train' if do_train else 'val'}_{out_file_time}.log"

    with open(os.path.join("defense/configs", f"exp{exp_num}a{advexp}_advtrain_{out_file_time}_{'R' if resume else ''}{'_val' if not do_train else ''}.config"), "w") as f:
        json.dump(config, f)
    with open(os.path.join(out_dir, f"exp{exp_num}a{advexp}_advtrain_{out_file_time}_{'R' if resume else ''}{'_val' if not do_train else ''}.config"), "w") as f:
        json.dump(config, f)


    with open(os.path.join(out_dir, out_file_name), "a+") as log_file:
        print(json.dumps(config, indent=4, sort_keys=True))
        print("\n".join(initialization_out), file=log_file)
        print(json.dumps(config, indent=4, sort_keys=True), file=log_file)
        for epoch in range(resume_epoch+1, 100):

            total_loss = 0

            att_model.train()
            acc_sum_train = 0
            acc_count_train = 0
            fp_sum_train = 0
            fp_count_train = 0

            fn_sum_train = 0
            fn_count_train = 0

            if do_train:
                for batch_idx, sample in enumerate(tqdm(train_loader)):

                    img = sample['img'].to(device)
                    orig_img = img.clone()
                    label = sample['label'].to(device)

                    with ctx_noparamgrad_and_eval(att_model):
                        data = adversary.perturb(img, label)

                        if one_way:
                            data[torch.flatten(label)==1] = orig_img[torch.flatten(label)==1]

                    outputs = att_model(data)
                    optimizer.zero_grad()

                    loss = criterion(outputs, label, gamma=adv_floss_gamma, alpha=adv_floss_alpha)

                    if isinstance(model, torch.nn.DataParallel):
                        loss = loss.sum()

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    classified_output = outputs > 0
                    equality = classified_output == label
                    accuracy = equality.sum().detach().cpu().numpy()
                    acc_sum_train += accuracy
                    count = equality.size()[0]
                    acc_count_train += count

                    inequality = classified_output != label
                    fp = label == 0
                    fp_inequality = inequality * fp
                    fp_rate = fp_inequality.sum().detach().cpu().numpy()
                    fp_sum_train += fp_rate
                    fp_rate_count = fp.sum().detach().cpu().numpy()
                    fp_count_train += fp_rate_count

                    fn = label == 1
                    fn_inequality = inequality * fn
                    fn_rate = fn_inequality.sum().detach().cpu().numpy()
                    fn_sum_train += fn_rate
                    fn_rate_count = fn.sum().detach().cpu().numpy()
                    fn_count_train += fn_rate_count

            if epoch % 1 == 0:
                save_dict = {
                    "epoch": epoch,
                    "net": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    "optim": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict()
                }

                save_name = os.path.join(out_dir, str(epoch) + '_advtrain.pth')
                torch.save(save_dict, save_name)


            att_model.eval()
            acc_sum = 0
            acc_count = 0

            fp_sum = 0
            fp_count = 0

            fn_sum = 0
            fn_count = 0
            total_loss_val = 0
            TP = 0
            for batch_idx, sample in enumerate(tqdm(val_loader)):

                img = sample['img'].to(device)
                orig_img = img.clone()
                label = sample['label'].to(device)

                outputs = []

                with ctx_noparamgrad_and_eval(att_model):
                    for _ in range(1):
                        data = adversary.perturb(img, label)

                        if one_way:
                            data[torch.flatten(label)==1] = orig_img[torch.flatten(label)==1]


                        output_single = att_model(data)
                        outputs.append(output_single)

                    outputs = torch.cat(outputs, dim=-1)
                    labelled_outputs = outputs > 0
                    outputs = torch.logical_xor(labelled_outputs, label).sum(1) > 0.

                acc_count += label.size()[0]
                acc_sum += int(label.size()[0] - outputs.sum().detach().cpu().numpy())



                inequality = outputs
                #
                fp = torch.flatten(label == 0)
                fp_inequality = inequality * fp
                fp_rate = fp_inequality.sum().detach().cpu().numpy()
                fp_sum += fp_rate
                fp_rate_count = fp.sum().detach().cpu().numpy()
                fp_count += fp_rate_count

                fn = torch.flatten(label == 1)
                fn_inequality = inequality * fn
                fn_rate = fn_inequality.sum().detach().cpu().numpy()
                fn_sum += fn_rate
                fn_rate_count = fn.sum().detach().cpu().numpy()
                fn_count += fn_rate_count

                print(f'{batch_idx} val raw: {acc_sum} {acc_count} {fp_sum} {fp_count} {fn_sum} {fn_count}')

            tensorboard.add_scalar("val/accuracy", float(acc_sum) / float(acc_count), epoch)
            tensorboard.add_scalar("val/fp", fp_sum/fp_count, epoch)
            tensorboard.add_scalar("val/fn", fn_sum/fn_count, epoch)

            if do_train:
                tensorboard.add_scalar("train/loss", total_loss/acc_count_train, epoch)
                tensorboard.add_scalar("train/accuracy", acc_sum_train/acc_count_train, epoch)
                tensorboard.add_scalar("train/fp", fp_sum_train/fp_count_train, epoch)
                tensorboard.add_scalar("train/fn", fn_sum_train/fn_count_train, epoch)


            result_str = f'Epoch {epoch}\n-----------\n' \
                  f'Train accuracy: {acc_sum_train/max(acc_count_train, 1)}\n' \
                         f'Train fp: {fp_sum_train/max(fp_count_train, 1)}\n' \
                         f'Train fn: {fn_sum_train/max(fn_count_train, 1)}\n' \
                         f'Val accuracy: {acc_sum / max(acc_count, 1):.4f}\n'\
                         f'Val fp: {fp_sum/max(fp_count, 1):.4f}\n'\
                         f'Val fn: {fn_sum/max(fn_count, 1):.4f}\n'\
                         f'Val raw: {acc_sum} {acc_count} {fp_sum} {fp_count} {fn_sum} {fn_count}\n'
            result_str += f"Time: {datetime.now()}"

            print(result_str, file=log_file)

            print(result_str)
            log_file.flush()

            if not do_train:
                exit()


