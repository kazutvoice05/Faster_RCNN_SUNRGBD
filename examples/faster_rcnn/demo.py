import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp
import glob
import chainer
from tqdm import tqdm

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import FasterRCNNVGG16
from chainercv import utils
from chainercv.visualizations import vis_bbox

import sys
sys.path.append(osp.curdir)
from SUNRGBD_dataset import SUNRGBDDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='voc07')
    parser.add_argument('--output-dir','-o', default='output_img')
    parser.add_argument('--images', '-i', default='/home/takagi.kazunari/projects/datasets/SUNRGBD_2DBB_fixed/test/image')
    args = parser.parse_args()

    dataset = SUNRGBDDataset("/home/takagi.kazunari/projects/datasets/SUNRGBD_2DBB_fixed")

    sunrgbd_bbox_label_names = dataset.get_dataset_label()

    model = FasterRCNNVGG16(n_fg_class=len(sunrgbd_bbox_label_names),pretrained_model=args.pretrained_model)
    if args.pretrained_model == 'voc07':
        model = FasterRCNNVGG16(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    f_names = glob.glob(osp.join(args.images, "*"))
    
    for f_name in tqdm(f_names):
        img = utils.read_image(osp.join(args.images, f_name), color=True)
        bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        out_f_name = f_name.split("/")[-1]
        vis_bbox(
            img, bbox, label, score, label_names=sunrgbd_bbox_label_names)
        plt.savefig(osp.join(args.output_dir, "res_" + out_f_name))


if __name__ == '__main__':
    main()
