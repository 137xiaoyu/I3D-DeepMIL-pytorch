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
parser.add_argument('--h5_file', default='E:/137/dataset/VAD/UCF_Crime/videos/h5py/UCF_Crime.h5', type=str)
parser.add_argument('--train_split', default='D:/137/workspace/python_projects/VAD_workspace/my_vad/misc/UCF_Crime_train.txt', type=str)
parser.add_argument('--test_split', default='D:/137/workspace/python_projects/VAD_workspace/my_vad/misc/UCF_Crime_test.txt', type=str)
parser.add_argument('--save_dir', default='E:/137/dataset/VAD/UCF_Crime/my_features/I3D/', type=str)

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

                    outputs = model.extract_features(inputs)  # (10, 1024, 1, 1, 1)
                    features.append(outputs.squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy())  # (n, 10, 1024)

                save_path = os.path.join(args.save_dir, status)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(os.path.join(save_path, video_name + '.npy'), np.stack(features, axis=0))  # (n, 10, 1024)


def main():
    if args.mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)

    i3d.load_state_dict(torch.load(args.ckpt))
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
