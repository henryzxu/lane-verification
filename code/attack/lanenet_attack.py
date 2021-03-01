
import torch.nn as nn
import torchvision.transforms.functional as F
import torch

from config import *
import dataset
from utils.transforms import *
from torch.utils.data import DataLoader



import os
import json
from matplotlib import pyplot as plt


from lane_proposal.model import LaneNet

os.chdir("../../")

# ------------ set config ------------
exp_dir = "./experiments/exp0"

# ------------ config ------------
exp_name = exp_dir.split('/')[-1]

with open(os.path.join(exp_dir, "cfg.json")) as f:
    exp_cfg = json.load(f)
resize_shape = tuple(exp_cfg['dataset']['resize_shape'])
device = torch.device("cuda:1")


# ------------ data prep ------------
dataset_name = exp_cfg['dataset'].pop('dataset_name')

Dataset_Type = getattr(dataset, dataset_name)

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

# ------------ val data [Note we use VAL Data] ------------
transform_val = Compose(Resize(resize_shape), ToTensor())

def build_data_loader(dset, batch_size=8, num_workers=4):
    val_dataset = Dataset_Type(Dataset_Path[dataset_name], dset, transform_val)
    lanenet_val_loader = DataLoader(val_dataset, pin_memory=True, batch_size=batch_size, collate_fn=val_dataset.collate, num_workers=num_workers)
    return lanenet_val_loader
# ------------ model preparation ------------
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

start_epoch = save_dict['epoch'] + 1

# Convert Model for attack
# turn Model into feature attack

class Lane_C(nn.Module):
    def __init__(
            self,
            base_model
    ):
        super(Lane_C, self).__init__()
        self.base = base_model


    def forward(self, img):

        img = torch.stack([F.normalize(i, mean, std) for i in img])
        x = self.base(img) # x is a dict
        return x['binary_seg']


att_model = Lane_C(net)
att_model = att_model.to(device)
att_model.eval()

lanenet_att_model = att_model

target_dir = r"C:\Users\henry\Dropbox\sp20\backup\henry\PycharmProjects\tmp\lane-detection\defense\attack_images_v2"
def get_targets(batch_size):
    selected_targets = np.random.choice(np.arange(len(os.listdir(target_dir))//3), batch_size)
    targets = []
    for idx in selected_targets:
        img_name = os.path.join(target_dir, f"{idx}_expanded.png")
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_CUBIC)
        img = torch.gt(torch.from_numpy(img), 0).type(torch.long)


        targets.append(img)

    return torch.stack(targets).to(device)



def save_image_from_raw_input(t, fname, permute=True, shift=0.0, scaling_factor=1, numpy_save=False):
    ext = ".png" if not numpy_save else ".npy"
    if fname[-4:] != ext:
        fname = fname + ext
    if permute:
        img = t.permute(1, 2, 0).numpy()
    else:
        img = t

    os.makedirs(os.path.split(fname)[0], exist_ok=True)
    if numpy_save:
        np.save(fname, img)
    else:
        img = scaling_factor*img + shift

        if np.count_nonzero(img > 1) > 1:
            img = img/255
        plt.imsave(fname, img)

def save_image_from_raw_bitmap(raw_binary_seg, fname, take_argmax=True):
    bin_seg_prob = raw_binary_seg.detach().cpu().numpy()
    if take_argmax:
        bin_seg_pred = np.argmax(bin_seg_prob, axis=0)
    else:
        bin_seg_pred = bin_seg_prob

    bin_seg_img = np.zeros((288, 512, 3), dtype=np.uint8)
    bin_seg_img[bin_seg_pred==1] = [1, 1, 1]
    bin_seg_img = cv2.cvtColor(bin_seg_img, cv2.COLOR_BGR2RGB)
    save_image_from_raw_input(bin_seg_img, fname, permute=False)


inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)


# TODO: change this to desired amount
NUM_SAMPLES = 5000


lanenet_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.4, 1.]).cuda(device=device))


def calc_iou(a, b):
    overlap = a * b  # Logical AND
    union = a + b  # Logical OR

    return np.count_nonzero(overlap) / float(np.count_nonzero(union))
