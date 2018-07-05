#coding: 'utf-8'

"""
faster_rcnn_chainercv
SUNRGBD_dataset

created by Kazunari on 2018/06/26 
"""

import numpy as np
import os
import warnings
import logging
import scipy.io
import glob

import chainer
from chainercv.utils import read_image

class SUNRGBDDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode

        with open(os.path.join(data_dir, str(self.mode), "{0}_list.txt").format(self.mode), "r") as f:
            id_list = f.readlines()

        for i in range(len(id_list)):
            id_list[i] = id_list[i].replace("\n", "")
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def get_example(self, i):
        if i >= len(self):
            raise IndexError("index is too large")

        id_ = self.id_list[i]

        matdata = scipy.io.loadmat(os.path.join(self.data_dir, str(self.mode), "annotation", "{0}.mat".format(id_)))

        bboxs = matdata["BBoxs"]

        for i in range(len(bboxs)):
            bboxs[i] = [bboxs[i][1],
                        bboxs[i][0],
                        bboxs[i][1] + bboxs[i][3] - 1,
                        bboxs[i][0] + bboxs[i][2] - 1]

        labels = matdata["class_ids"]
        if len(labels) != 0:
            labels = labels[0]

            # 1 ~ n -> 0 ~ n -1
            for i in range(len(labels)):
                labels[i] = labels[i] - 1

        if len(bboxs) == 0:
            bboxs = [[]]
        labels = np.asarray(labels, dtype=np.int32)
        bboxs = np.asarray(bboxs, dtype=np.float32)

        img_file = os.path.join(
            self.data_dir, str(self.mode), "image", "{0}.png".format(id_))
        img = read_image(img_file, color=True)

        return img, bboxs, labels

    def get_dataset_label(self):
        with open(os.path.join(self.data_dir, "class_ids.txt"), "r") as f:
            lines = f.readlines()

        classes = []
        for line in lines:
            class_name = line.split(":")[1]
            class_name = class_name.replace("\n","")
            classes.append(class_name)

        return tuple(classes)
