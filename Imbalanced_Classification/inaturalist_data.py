from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import json
import numpy as np


class iNaturalist(data.Dataset):
    """Dataset class for the inaturalist dataset."""

    def __init__(self, image_dir, transform, mode='train'):
        """Initialize and preprocess the inaturalist dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.train_json = "./train2018.json"
        self.val_json = "./val2018.json"
        self.test_json = "./test2018.json"
        self.cls_num_list = [0]*8142
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        elif mode == "val":
            self.num_images = len(self.val_dataset)
        else: 
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the json file."""
        train_dict = json.load(open(self.train_json, "r"))
        val_dict = json.load(open(self.val_json, "r"))
        test_dict = json.load(open(self.test_json, "r"))

        for i in range(0, len(train_dict["images"])):
            filename = train_dict["images"][i]["file_name"]      
            label = train_dict["annotations"][i]["category_id"]
            self.train_dataset.append([filename, label])
            self.cls_num_list[int(label)] += 1

        for i in range(0, len(val_dict["images"])):
            filename = val_dict["images"][i]["file_name"]
            label = val_dict["annotations"][i]["category_id"]
            self.val_dataset.append([filename, label])

        for i in range(0, len(test_dict["images"])):
            filename = test_dict["images"][i]["file_name"]
            img_id = test_dict["images"][i]["id"]
            self.test_dataset.append([filename, img_id])


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if self.mode == "train":
           dataset = self.train_dataset
        elif self.mode == "val":
           dataset = self.val_dataset
        else:
           dataset = self.test_dataset

        if self.mode == "test":
           filename, img_id = dataset[index]
           image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
           return self.transform(image), img_id
        else:
           filename, label = dataset[index]
           image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
           return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return self.num_images

    def get_cls_num_list(self):
        return self.cls_num_list

