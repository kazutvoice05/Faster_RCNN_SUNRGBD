"""
faster_rcnn_chainercv
train_SUNRGBD

created by Kazunari on 2018/06/26 
"""

from __future__ import division

import argparse
import numpy as np
import os.path as osp
import datetime

import matplotlib
matplotlib.use('Agg')

import chainer
from chainer.datasets import TransformDataset
from chainer import training
from chainer.training import extensions
from chainer.training.triggers import ManualScheduleTrigger

from chainercv.datasets import voc_bbox_label_names
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from chainercv import transforms

import sys
sys.path.append(osp.curdir)
from SUNRGBD_dataset import SUNRGBDDataset


class Transform(object):

    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


def main():
    parser = argparse.ArgumentParser(
        description='ChainerCV training example: Faster R-CNN')
    parser.add_argument('--dataset_path', '-path', type=str, default="/home/takagi.kazunari/projects/datasets/SUNRGBD_2DBB_fixed")
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--out', '-o', default='sunrgbd_result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--step_size', '-ss', type=int, default=50000)
    parser.add_argument('--iteration', '-i', type=int, default=70000)
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_data = SUNRGBDDataset(args.dataset_path,mode="train")
    test_data = SUNRGBDDataset(args.dataset_path, mode="test")

    sunrgbd_bbox_label_names = train_data.get_dataset_label()

    faster_rcnn = FasterRCNNVGG16(n_fg_class=len(sunrgbd_bbox_label_names),
                                  pretrained_model='imagenet')
    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))

    train_data = TransformDataset(train_data, Transform(faster_rcnn))

    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size=1, n_processes=None, shared_mem=100000000)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)

    now_time = str(datetime.datetime.today()).replace(" ","_")
    save_dir = osp.join(args.out, now_time)

    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=save_dir)

    #save_iteration = [i for i in range(100, args.iteration, args.step_size)]

    weight_save_interval = 5000, 'iteration'
    evaluation_interval = 10000, 'iteration'

    trainer.extend(
        extensions.snapshot_object(model.faster_rcnn, 'sunrgbd_model_{.updater.iteration}.npz'),
        trigger=weight_save_interval)
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.step_size, 'iteration'))

    log_interval = 20, 'iteration'
    plot_interval = 10, 'iteration'
    print_interval = 20, 'iteration'

    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/map',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss'],
                file_name='loss.png', trigger=plot_interval
            ),
            trigger=plot_interval
        )

    #do_evaluation_iteration = [i for i in range(0, args.iteration, 500)]

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model.faster_rcnn, use_07_metric=True,
            label_names=sunrgbd_bbox_label_names),
        trigger=evaluation_interval)

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()


if __name__ == '__main__':
    main()
