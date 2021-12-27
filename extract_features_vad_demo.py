import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('-load_model', default='models/rgb_imagenet.pt', type=str)
parser.add_argument('-root', default='D:/137/dataset/action_recognition/Charades_v1_rgb/', type=str)
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-save_dir', default='I3D_features/', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset
import cv2


def load_rgb_frames(image_dir, start):
    num_frames = len(os.listdir(image_dir))
    frames = []
    for i in range(start, start + num_frames):
        img = cv2.imread(os.path.join(image_dir, str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 256 or h < 256:
            d = 256. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def run(max_steps=64e3,
        mode='rgb',
        root='/ssd2/charades/Charades_v1_rgb',
        split='charades/charades.json',
        batch_size=1,
        load_model='',
        save_dir=''):
    # setup dataset
    name = 'Abuse001_x264'
    img_dir = f'D:/Users/Chase/Desktop/{name}/'
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    i3d.eval()  # Set model to evaluate mode

    with torch.no_grad():
        # get the inputs
        inputs = load_rgb_frames(img_dir, 1)  # t, h, w, c
        inputs = test_transforms(inputs)
        inputs = torch.from_numpy(inputs.transpose([3, 0, 1, 2])).unsqueeze(0)  # b, c, t, h, w

        b, c, t, h, w = inputs.shape

        features = []
        for start in range(0, t, 16):
            end = min(t, start + 16)
            if end - start < 16:
                break
            ip = torch.from_numpy(inputs.numpy()[:, :, start:end]).cuda()
            features.append(
                i3d.extract_features(ip).squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy())

    np.save(os.path.join(save_dir, name), np.concatenate(features, axis=0).squeeze((1, 2)))


if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
