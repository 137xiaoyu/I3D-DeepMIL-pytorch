import argparse
import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import videotransforms
from pytorch_i3d import InceptionI3d

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--ckpt', default='models/rgb_imagenet.pt', type=str)
parser.add_argument('--h5_file', default='E:/137/dataset/VAD/XD_Violence/videos/h5py/XD_Violence.h5', type=str)
parser.add_argument('--train_split', default='D:/137/workspace/python_projects/VAD_workspace/my_vad/misc/XD_Violence_train.txt', type=str)
parser.add_argument('--test_split', default='D:/137/workspace/python_projects/VAD_workspace/my_vad/misc/XD_Violence_test.txt', type=str)
parser.add_argument('--save_dir', default='E:/137/dataset/VAD/XD_Violence/features/I3D/', type=str)

args = parser.parse_args()


def decode_frames(raw_frames, shorter_side, transforms):
    frames = []
    for frame in raw_frames:
        img = cv2.cvtColor(cv2.imdecode(frame, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        if h < shorter_side or w < shorter_side:
            d = float(shorter_side) - min(h, w)
            sc = 1 + d / min(h, w)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)

    frames = np.asarray(frames, dtype=np.float32)
    if transforms is not None:
        frames = transforms(frames)
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

                    frames = decode_frames(raw_frames, 256, transforms)  # t h w c

                    inputs = torch.from_numpy(frames.transpose([3, 0, 1, 2])).unsqueeze(0)  # b, c, t, h, w
                    inputs = inputs.cuda()

                    outputs = model.extract_features(inputs)
                    features.append(outputs.squeeze(0).permute(1, 2, 3, 0).cpu().numpy())

                save_path = os.path.join(args.save_dir, status)
                if not os.path.exists(os.path.join(save_path, video_name.split('/')[0])):
                    os.makedirs(os.path.join(save_path, video_name.split('/')[0]))

                np.save(os.path.join(save_path, video_name + '.npy'), np.concatenate(features, axis=0).squeeze((1, 2)))


def main():
    if args.mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    i3d.load_state_dict(torch.load(args.ckpt))
    i3d.cuda()

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    with open(args.train_split, 'r') as txt_file:
        train_video_list = txt_file.readlines()

    with open(args.test_split, 'r') as txt_file:
        test_video_list = txt_file.readlines()

    save_features(i3d, train_video_list, test_transforms, train=True)
    save_features(i3d, test_video_list, test_transforms, train=False)


if __name__ == '__main__':
    main()
