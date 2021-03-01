import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle as pkl
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image

class DefenseDataset(Dataset):


    def __init__(self, path, image_set, limit=0, transforms=None, ext=".png"):
        super(DefenseDataset, self).__init__()
        # assert image_set in ('train', 'val', "val_resize_v1", "data_resize_v1", "data_sr1_v1", "val_sr1_v1", "data_grounded_fixed", "data_grounded_fixed_v3", "data_grounded_fixed_v7", "val_grounded_fixed",
        #                      "val_grounded_fixed_v3", "test_grounded_fixed_v3", "val_grounded_fixed_v7",
        #                      'test', 'test_grounded', 'test_v2', 'custom',
        #                      'random', "val_random", "data_grounded", "val_random_grounded"), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.limit = limit

        self.ext = ext

        self.createIndex()

    def __len__(self):
        return len(self.img_list)

    def createIndex(self):
        self.img_list = []
        self.label_list = []
        if self.limit:
            listfile = os.path.join(self.data_dir_path, "{}_gt_{}.txt".format(self.image_set, self.limit))
            print(listfile)

            with open(listfile, "rb") as f:
                lines = pkl.load(f)
                for line in lines:
                    for filename in os.listdir(os.path.join(line["defense_dir"], "positive")):
                        self.img_list.append(os.path.join(os.path.join(line["defense_dir"], "positive"), filename))
                        self.label_list.append(1)

                    for filename in os.listdir(os.path.join(line["defense_dir"], "negative")):
                        self.img_list.append(os.path.join(os.path.join(line["defense_dir"], "negative"),
                                                          filename))
                        self.label_list.append(0)
        else:
            if self.image_set == "train":
                data_folders = ["data_v2", "data_v2_p2"]
            elif self.image_set == "val":
                data_folders = ["data_val_v2"]
            elif self.image_set == "test":
                data_folders = ["data_test"]
            elif self.image_set == "test_v2":
                data_folders = ["data_test_v2"]
            elif self.image_set == "custom":
                data_folders = ["data_custom"]
            elif self.image_set == "random":
                data_folders = ["data_random_0313"]
            elif self.image_set == "val_random":
                data_folders = ["data_val_0531"]
            elif self.image_set == "data_grounded":
                data_folders = ["data_random_0313_grounded"]
            elif self.image_set == "val_random_grounded":
                data_folders = ["data_val_grounded_0531"]
            elif self.image_set == "test_grounded":
                data_folders = ["test_grounded"]
            elif self.image_set == "data_grounded_fixed":
                data_folders = ["data_horizontal_fixed"]
            elif self.image_set == "val_grounded_fixed":
                data_folders = ["data_val_horizontal_fixed"]
            elif self.image_set == "data_grounded_fixed_v3":
                data_folders = ["data_horizontal_fixed_v3"]
            elif self.image_set == "val_grounded_fixed_v3":
                data_folders = ["data_val_horizontal_fixed_v3"]

            elif self.image_set == "data_grounded_fixed_v7":
                data_folders = ["data_horizontal_fixed_v7"]
            elif self.image_set == "val_grounded_fixed_v7":
                data_folders = ["data_val_horizontal_fixed_v7"]


            elif self.image_set == "data_sr1_v1":
                data_folders = ["data_sr1_v1"]
            elif self.image_set == "val_sr1_v1":
                data_folders = ["data_val_sr1_v1"]

            elif self.image_set == "val_resize_v1":
                data_folders = ["val_resize_v1"]

            elif self.image_set == "data_resize_v1":
                data_folders = ["data_resize_v1"]

            elif self.image_set == "val_resize_sr0.5_v1":
                data_folders = ["val_resize_sr0.5_v1"]
            elif self.image_set == "val_resize_sr1_interarea_v1":
                data_folders = ["val_resize_sr1_interarea_v1"]
            elif self.image_set == "train_resize_sr1_interarea_v1":
                data_folders = ["train_resize_sr1_interarea_v1"]

            elif self.image_set == "test_grounded_fixed_v3":
                data_folders = ["data_test_horizontal_fixed_v3"]
            else:
                data_folders = [self.image_set]
            total_neg = 0
            total_pos = 0

            for fdata in data_folders:
                for idx, folder in enumerate(tqdm(sorted(os.listdir(os.path.join(self.data_dir_path, fdata))))):
                    defense_dir = os.path.join(self.data_dir_path, fdata, folder)
                    # print(defense_dir)
                    pos_count = 0
                    for filename in os.listdir(os.path.join(defense_dir, "positive")):
                        if filename[-4:] == self.ext:
                            self.img_list.append(os.path.join(os.path.join(defense_dir, "positive"), filename))
                            self.label_list.append(1)
                            pos_count += 1

                    total_pos += pos_count

                    neg_count = 0
                    for filename in os.listdir(os.path.join(defense_dir, "negative")):
                        if filename[-4:]==self.ext:
                            self.img_list.append(os.path.join(os.path.join(defense_dir, "negative"),
                                                              filename))
                            self.label_list.append(0)
                            neg_count += 1
                            # if neg_count >= pos_count:
                            #     break
                    total_neg += neg_count

            print(total_neg/total_pos)


    def __getitem__(self, idx):
        if self.ext==".png":
            img = cv2.imread(self.img_list[idx])
            # img = img[:, 10:-10, :]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.ext==".npy":
            img = np.load(self.img_list[idx], allow_pickle=True)
        # img = img.ravel()
        #
        # img = torch.from_numpy(img).type(torch.float32) / 255.



        label = self.label_list[idx]
        label = torch.from_numpy(np.array([label])).type(torch.float32)


        sample = {'img': img,
                  'label': label,
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['label'] is None:
            label = None
        elif isinstance(batch[0]['label'], torch.Tensor):
            label = torch.stack([b['label'] for b in batch])
        else:
            label = [b['label'] for b in batch]

        samples = {'img': img,
                   'label': label,
                   'img_name': [x['img_name'] for x in batch]}

        return samples




class DefensePatchDataset(Dataset):


    def __init__(self, path, image_set, limit=0, transforms=None, patch_height=30, min_patch=5, ext=".png", interpolation=cv2.INTER_AREA):
        super(DefensePatchDataset, self).__init__()
        # assert image_set in ('train', 'val', "val_resize_v1", "data_resize_v1", "data_sr1_v1", "val_sr1_v1", "data_grounded_fixed", "data_grounded_fixed_v3", "data_grounded_fixed_v7", "val_grounded_fixed",
        #                      "val_grounded_fixed_v3", "test_grounded_fixed_v3", "val_grounded_fixed_v7",
        #                      'test', 'test_grounded', 'test_v2', 'custom',
        #                      'random', "val_random", "data_grounded", "val_random_grounded"), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.limit = limit

        self.ext = ext
        self.patch_height = patch_height
        self.min_height = patch_height * min_patch

        self.interpolation = interpolation

        self.createIndex()

    def __len__(self):
        return len(self.img_list)

    def createIndex(self):
        self.img_list = []
        self.label_list = []
        self.patch_idx = []

        if self.image_set == "val_resize_sr1_interarea_v1":
            data_folders = ["val_resize_sr1_interarea_v1"]
        elif self.image_set == "train_resize_sr1_interarea_v1":
            data_folders = ["train_resize_sr1_interarea_v1"]

        else:
            raise AssertionError("image set not correct")
        total_neg = 0
        total_pos = 0

        for fdata in data_folders:
            for idx, folder in enumerate(sorted(os.listdir(os.path.join(self.data_dir_path, fdata)))):
                defense_dir = os.path.join(self.data_dir_path, fdata, folder)
                # print(defense_dir)
                classes = ["negative", "positive"]

                for class_idx in range(len(classes)):
                    class_name = classes[class_idx]


                    for filename in os.listdir(os.path.join(defense_dir, class_name)):
                        if filename[-4:] == self.ext:
                            abs_loc = os.path.join(os.path.join(defense_dir, class_name), filename)
                            loaded_img = self.load_image(abs_loc)
                            loaded_img_height = max(loaded_img.shape[0], self.min_height)
                            num_patches = (int(loaded_img_height // (self.patch_height)) - 1) * 3
                            self.img_list.extend([abs_loc] * num_patches)
                            self.label_list.extend([class_idx] * num_patches)
                            self.patch_idx.extend(list(range(num_patches)))




                        # if neg_count >= pos_count:
                        #     break




    def load_image(self, loc):
        if self.ext==".png":
            img = cv2.imread(loc)
            # img = img[:, 10:-10, :]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.ext==".npy":
            img = np.load(loc, allow_pickle=True)

        return img




    def __getitem__(self, idx):
        img = self.load_image(self.img_list[idx])
        # img = img.ravel()
        #
        # img = torch.from_numpy(img).type(torch.float32) / 255.






        label = self.label_list[idx]
        label = torch.from_numpy(np.array([label])).type(torch.float32)


        sample = {'img': img,
                  'label': label,
                  'patch_idx': self.patch_idx[idx],
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['label'] is None:
            label = None
        elif isinstance(batch[0]['label'], torch.Tensor):
            label = torch.stack([b['label'] for b in batch])
        else:
            label = [b['label'] for b in batch]

        samples = {'img': img,
                   'label': label,
                   'patch_idx': [x['patch_idx'] for x in batch],
                   'img_name': [x['img_name'] for x in batch]}

        return samples



class DefensePatchDatasetV2(Dataset):


    def __init__(self, path, image_set, limit=0, transforms=None,
                 patch_height=60, min_patch=5, ext=".png",
                 patch_offset=20,
                 interpolation=cv2.INTER_AREA):
        super(DefensePatchDatasetV2, self).__init__()
        # assert image_set in ('train', 'val', "val_resize_v1", "data_resize_v1", "data_sr1_v1", "val_sr1_v1", "data_grounded_fixed", "data_grounded_fixed_v3", "data_grounded_fixed_v7", "val_grounded_fixed",
        #                      "val_grounded_fixed_v3", "test_grounded_fixed_v3", "val_grounded_fixed_v7",
        #                      'test', 'test_grounded', 'test_v2', 'custom',
        #                      'random', "val_random", "data_grounded", "val_random_grounded"), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms
        self.limit = limit

        self.ext = ext
        self.patch_height = patch_height
        self.min_height = patch_height * min_patch
        self.patch_offset = patch_offset

        self.interpolation = interpolation

        self.createIndex()

    def __len__(self):
        return len(self.img_list)

    def createIndex(self):
        self.img_list = []
        self.label_list = []
        self.patch_idx = []

        if self.image_set == "val_resize_sr1_interarea_v1":
            data_folders = ["val_resize_sr1_interarea_v1"]
        elif self.image_set == "train_resize_sr1_interarea_v1":
            data_folders = ["train_resize_sr1_interarea_v1"]

        else:
            raise AssertionError("image set not correct")
        total_neg = 0
        total_pos = 0

        for fdata in data_folders:
            for idx, folder in enumerate(tqdm(sorted(os.listdir(os.path.join(self.data_dir_path, fdata))))):
                defense_dir = os.path.join(self.data_dir_path, fdata, folder)
                # print(defense_dir)
                classes = ["negative", "positive"]

                for class_idx in range(len(classes)):
                    class_name = classes[class_idx]


                    for filename in os.listdir(os.path.join(defense_dir, class_name)):
                        if filename[-4:] == self.ext:
                            abs_loc = os.path.join(os.path.join(defense_dir, class_name), filename)
                            loaded_img = self.load_image(abs_loc)
                            loaded_img_height = max(loaded_img.shape[0], self.min_height)
                            patch_count = 0
                            while loaded_img_height - self.patch_height >= 0:
                                patch_count += 1
                                loaded_img_height -= self.patch_offset

                            # print(filename, patch_count)

                            self.img_list.extend([abs_loc] * patch_count)
                            self.label_list.extend([class_idx] * patch_count)
                            self.patch_idx.extend(list(range(patch_count)))




                        # if neg_count >= pos_count:
                        #     break




    def load_image(self, loc):
        if self.ext==".png":
            img = cv2.imread(loc)
            # img = img[:, 10:-10, :]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.ext==".npy":
            img = np.load(loc, allow_pickle=True)

        return img




    def __getitem__(self, idx):
        img = self.load_image(self.img_list[idx])
        # img = img.ravel()
        #
        # img = torch.from_numpy(img).type(torch.float32) / 255.






        label = self.label_list[idx]
        label = torch.from_numpy(np.array([label])).type(torch.float32)


        sample = {'img': img,
                  'label': label,
                  'patch_idx': self.patch_idx[idx],
                  'img_name': self.img_list[idx]}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['label'] is None:
            label = None
        elif isinstance(batch[0]['label'], torch.Tensor):
            label = torch.stack([b['label'] for b in batch])
        else:
            label = [b['label'] for b in batch]

        samples = {'img': img,
                   'label': label,
                   'patch_idx': [x['patch_idx'] for x in batch],
                   'img_name': [x['img_name'] for x in batch]}

        return samples

