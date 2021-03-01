import json
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from verifier.defense_dataset import DefenseDataset
from torch.utils.data import DataLoader

from verifier.defense_nn_models import DefenseMLP, DefenseLinear, DefenseMaxPool
from utils.transforms import *
import os

from datetime import datetime

os.chdir("../../")

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

resize_shape = (40,128)
kernel_size = 3
exp_num = 1500

hidden_layer1 = 32
hidden_layer2 = 128

model_type="conv"

additional_modifiers = '_ord2'

sample_rate = 1

noise_sd = 0

resume_epoch = 20

do_train = True

f_gamma = 1
f_alpha = 0.75
bce_pos_weight = 1

exp = f"exp{exp_num}"

train_dataset = "train_resize_sr1_ord2"
val_dataset = "train_resize_sr1_ord2"
dataset_ext = ".npy"

interpolation = "INTER_AREA"
interpolation_cv2 = cv2.INTER_AREA
tensorboard = SummaryWriter(log_dir="defense/logs/"+exp)



settings_dict = {
    "mean": mean,
    "std": std,
    "resize_shape": resize_shape,
    "kernel_size": kernel_size,
    "exp_num": exp_num,
    "exp": exp,
    "hidden_layer1": hidden_layer1,
    "hidden_layer2": hidden_layer2,
    "model_type": model_type,
    "additional_modifiers": additional_modifiers,
    "noise_sd": noise_sd,
    "resume_epoch": resume_epoch,
    "do_train": do_train,
    "timestamp": f"{datetime.now()}",
    "floss_gamma": f_gamma,
    "floss_alpha": f_alpha,
    "sample_rate": sample_rate,
    "train_dataset": train_dataset,
    "val_dataset": val_dataset,
    "dataset_ext": dataset_ext,
    "bce_pos_weight": bce_pos_weight,
    "interpolation": interpolation,
    "interpolation_cv2": interpolation_cv2
}


if __name__=='__main__':
    transform_x = Compose(Resize(resize_shape, interpolation=interpolation_cv2), ToTensor(), Normalize(mean=mean, std=std))
    train_loader = DataLoader(DefenseDataset("defense", train_dataset, transforms=transform_x, ext=dataset_ext), batch_size=128, shuffle=True, collate_fn=DefenseDataset.collate)

    val_loader = DataLoader(DefenseDataset("defense", val_dataset, transforms=transform_x, ext=dataset_ext), batch_size=128, shuffle=True, collate_fn=DefenseDataset.collate)


    device = torch.device("cuda:1")


    if model_type == "conv":
        model =  DefenseMLP(kernel_size=kernel_size, input_shape=resize_shape).to(device)
    elif model_type =="conv_maxpool":
        model =  DefenseMaxPool(kernel_size=kernel_size, input_shape=resize_shape).to(device)
    else:
        model = DefenseLinear(3*np.prod(resize_shape), hidden_layer1, hidden_layer2).to(device)

    save_name = os.path.join("defense", "saved_models", exp, str(resume_epoch) + '.pth')
    if not os.path.exists(save_name):
        resume_epoch = 0
        if os.path.exists(os.path.join("defense", "saved_models", exp)):
            raise AssertionError("existing experiment!")
    else:
        print("loading {}".format(save_name))

        save_dict = torch.load(save_name)
        model.load_state_dict(save_dict['net'])

    settings_dict["revised_epoch"] = resume_epoch


    def criterion(predictions, targets, gamma=2, alpha=0.75):
        if model_type == 'baseline':
            return torch.nn.MSELoss()(predictions, targets)
        bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones(1).cuda(device=device) * bce_pos_weight)(predictions, targets)
        alpha_tensor = (1-alpha) + targets * (2 * alpha - 1)
        p_t = torch.exp(-bce_loss)
        focal_loss = alpha_tensor * torch.pow(1-p_t, gamma) * bce_loss
        return focal_loss.mean()

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3, lr=1e-3)

    # summary(norm_model, (3, resize_shape[0], resize_shape[1]))

    out_dir = os.path.join("defense", "saved_models", exp)
    os.makedirs(out_dir, exist_ok=True)

    config_file_name = time.strftime("%Y-%m-%d__%H-%M-%S")
    with open(os.path.join("defense/configs", f"exp{exp_num}_nntrain_{config_file_name}.config"), "w") as f:
        json.dump(settings_dict, f)
    with open(os.path.join(out_dir, f"exp{exp_num}_nntrain_{config_file_name}.config"), "w") as f:
        json.dump(settings_dict, f)

    print(json.dumps(settings_dict, indent=4, sort_keys=True))

    with open(os.path.join(out_dir, f"exp{exp_num}_nntrain_{config_file_name}.log"), "a") as log_file:
        print(json.dumps(settings_dict, indent=4, sort_keys=True), file=log_file)
        for epoch in range(resume_epoch + 1, 100):
            total_loss = 0

            model.train()
            acc_sum_train = 0
            acc_count_train = 0
            fp_sum_train = 0
            fp_count_train = 0

            fn_sum_train = 0
            fn_count_train = 0

            if do_train:
                for batch_idx, sample in enumerate(tqdm(train_loader)):
                    optimizer.zero_grad()
                    img = sample['img'].to(device)
                    img = img + torch.randn_like(img, device='cuda:1') * noise_sd
                    label = sample['label'].to(device)


                    outputs = model(img)


                    loss = criterion(outputs, label, gamma=f_gamma, alpha=f_alpha)

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





            model.eval()
            acc_sum = 0
            acc_count = 0

            fp_sum = 0
            fp_count = 0

            fn_sum = 0
            fn_count = 0
            total_loss_val = 0
            TP = 0

            inner_loop = 1 if noise_sd == 0 else 10
            with torch.no_grad():
                for _ in tqdm(range(inner_loop)):
                    for batch_idx, sample in enumerate(val_loader):
                        img = sample['img'].to(device)

                        img = img + torch.randn_like(img, device='cuda:1') * noise_sd
                        label = sample['label'].to(device)


                        outputs = model(img)

                        loss = criterion(outputs, label, gamma=f_gamma, alpha=f_alpha)

                        if isinstance(model, torch.nn.DataParallel):
                            loss = loss.sum()

                        total_loss_val += loss.item()

                        classified_output = outputs > 0
                        equality = classified_output == label
                        accuracy = equality.sum().detach().cpu().numpy()
                        acc_sum += accuracy
                        count = equality.size()[0]
                        acc_count += count

                        inequality = classified_output != label

                        fp = label == 0
                        fp_inequality = inequality * fp
                        fp_rate = fp_inequality.sum().detach().cpu().numpy()
                        fp_sum += fp_rate
                        fp_rate_count = fp.sum().detach().cpu().numpy()
                        fp_count += fp_rate_count

                        fn = label == 1
                        fn_inequality = inequality * fn
                        fn_rate = fn_inequality.sum().detach().cpu().numpy()
                        fn_sum += fn_rate
                        fn_rate_count = fn.sum().detach().cpu().numpy()
                        fn_count += fn_rate_count

                        TP += (equality * fn).sum().detach().cpu().numpy()

            tensorboard.add_scalar("val/loss", total_loss_val/acc_count, epoch)
            tensorboard.add_scalar("val/accuracy", acc_sum / acc_count, epoch)
            tensorboard.add_scalar("val/fp", fp_sum/fp_count, epoch)
            tensorboard.add_scalar("val/fn", fn_sum/fn_count, epoch)
            tensorboard.add_scalar("val/precision", TP / (fp_sum + TP), epoch)
            tensorboard.add_scalar("val/recall", TP / (fn_sum + TP), epoch)
            tensorboard.add_scalar("val/f1", 2 / ((TP / (fn_sum + TP)) ** (-1) + (TP / (fp_sum + TP)) ** (-1)), epoch)


            if do_train:
                tensorboard.add_scalar("train/loss", total_loss/acc_count_train, epoch)
                tensorboard.add_scalar("train/accuracy", acc_sum_train/acc_count_train, epoch)
                tensorboard.add_scalar("train/fp", fp_sum_train/fp_count_train, epoch)
                tensorboard.add_scalar("train/fn", fn_sum_train/fn_count_train, epoch)

                result_str = f'Epoch {epoch}\n-----------\nTraining loss: {total_loss/acc_count_train:.4f} Training accuracy: {acc_sum_train/acc_count_train:.4f}'\
                  f' Train FP: {fp_sum_train/fp_count_train:4f} Train FN: {fn_sum_train/fn_count_train:4f}\n'
            else:
                result_str = f'Epoch {epoch}\n-----------\n'

            result_str +=  f'Val loss: {total_loss_val/acc_count:.4f} Val accuracy: {acc_sum / acc_count:.4f} Val FP: {fp_sum/fp_count:4f}'\
                  f' Val FN: {fn_sum/fn_count}\n'

            result_str += f"Date: {datetime.now()}\n==============\n"

            print(result_str, file=log_file)

            print(result_str)
            log_file.flush()

            if epoch % 10 == 0 and do_train:
                save_dict = {
                    "epoch": epoch,
                    "net": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    "optim": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict()
                }

                save_name = os.path.join(out_dir, str(epoch) + '.pth')
                torch.save(save_dict, save_name)

            if not do_train:
                exit()

        print("train finished at {}".format(datetime.now()), file=log_file)






