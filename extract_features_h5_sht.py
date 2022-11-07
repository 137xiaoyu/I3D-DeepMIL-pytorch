import argparse
import os

import cv2
import h5py
import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import videotransforms
from Model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--ckpt', default='vad_ckpt/checkpoint_0015.pth.tar', type=str)
parser.add_argument('--h5_file', default='D:/137/dataset/VAD/ShanghaiTech/videos/h5py/ShanghaiTech.h5', type=str)
parser.add_argument('--train_split', default='D:/137/workspace/python_projects/VAD_workspace/my_vad/misc/ShanghaiTech_train.txt', type=str)
parser.add_argument('--test_split', default='D:/137/workspace/python_projects/VAD_workspace/my_vad/misc/ShanghaiTech_test.txt', type=str)
parser.add_argument('--save_dir', default='D:/137/dataset/VAD/ShanghaiTech/my_features/I3D-10crop/', type=str)

args = parser.parse_args()


def decode_frames(raw_frames, shorter_side, transforms):
    frames = []
    for frame in raw_frames:
        img = cv2.cvtColor(cv2.imdecode(frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        d = float(shorter_side) - min(h, w)
        sc = 1 + d / min(h, w)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        assert min(img.shape[:2]) == shorter_side
        frames.append(img)

    frames = torch.from_numpy(np.asarray(frames, dtype=np.float32)).permute(0, 3, 1, 2)  # (t, h, w, c) -> (t, c, h, w)
    if transforms is not None:
        frames = transforms(frames).permute(0, 2, 1, 3, 4)  # (10, t, c, h, w) -> (10, c, t, h, w)
    return frames


def save_features(model, video_list, transforms, train=False):
    status = 'Train' if train else 'Test'
    model.eval()
    with torch.no_grad():
        with h5py.File(args.h5_file, 'r') as h5_file:
            for video in tqdm(video_list, desc=f'{status} video'):
                features = []

                video_name, num_frames = os.path.split(video.strip())
                seg_num = int(num_frames) // 16
                idx_all = list(np.arange(seg_num))

                for idx in idx_all:
                    raw_frames = []
                    for i in range(16):
                        key = '{:06d}'.format(idx * 16 + i)
                        raw_frames.append(h5_file[video_name][key][:])

                    frames = decode_frames(raw_frames, 256, transforms)  # 10, c, t, h, w
                    inputs = frames.cuda()  # 10, c, t, h, w

                    logits, outputs = model(inputs)  # (10, 2), (10, 1024)
                    features.append(outputs.cpu().numpy())  # (n, 10, 1024)

                save_path = os.path.join(args.save_dir, status)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(os.path.join(save_path, video_name + '.npy'), np.stack(features, axis=0))  # (n, 10, 1024)


def main():
    i3d = Model(2)

    checkpoint = torch.load(args.ckpt)

    new_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'queue' in k:
            continue
        new_dict[k] = v

    i3d.load_state_dict(new_dict)
    i3d.cuda()

    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_transforms = transforms.Compose([transforms.TenCrop(224), transforms.Lambda(lambda crops: torch.stack(crops))])

    with open(args.train_split, 'r') as txt_file:
        train_video_list = txt_file.readlines()

    with open(args.test_split, 'r') as txt_file:
        test_video_list = txt_file.readlines()

    save_features(i3d, train_video_list, test_transforms, train=True)
    save_features(i3d, test_video_list, test_transforms, train=False)


if __name__ == '__main__':
    main()
