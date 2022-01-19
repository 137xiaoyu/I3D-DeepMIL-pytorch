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
parser.add_argument('--train_split',
                    default='E:/137/dataset/VAD/UCF_Crime/videos/h5py/UCF_Crime_train.txt',
                    type=str)
parser.add_argument('--test_split',
                    default='E:/137/dataset/VAD/UCF_Crime/videos/h5py/UCF_Crime_test.txt',
                    type=str)
parser.add_argument('--save_dir', default='E:/137/dataset/VAD/UCF_Crime/I3D_features/', type=str)
parser.add_argument('--smaller_size', default=256, type=int)
parser.add_argument('--crop_size', default=224, type=int)

args = parser.parse_args()


class Extract_UCF_Dataset(Dataset):

    def __init__(self, h5_file, split_file, smaller_size=256, crop_size=224):
        super().__init__()
        self.h5_file = h5_file
        self.keys = self.read_split(split_file)
        self.smaller_size = smaller_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        """Not finished
        
        Cannot read a complete video due to memory limits
        While saving clip features may be troublesome
        """
        key = self.keys(index)
        with h5py.File(self.h5_file, 'r') as h5_file:
            frames = h5_file[key][:]

    def read_split(self, split_file):
        with open(split_file, 'r') as txt_file:
            video_list = txt_file.readlines()

        keys = []

        for video in video_list:
            video_class, video_info = os.path.split(video.strip())
            video_name, seg_num = video_info.rsplit('_', maxsplit=1)
            idx_all = list(np.arange(int(seg_num)))
            keys.extend(
                list(
                    map(lambda idx: f'{os.path.splitext(os.path.basename(video_name))[0]}_{idx:06d}',
                        idx_all)))

        return keys


def decode_frames(raw_frames, smaller_size):
    frames = []
    for frame in raw_frames:
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        if h < smaller_size or w < smaller_size:
            d = float(smaller_size) - min(h, w)
            sc = 1 + d / min(h, w)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def save_features(model, video_list, transforms, train=False):
    status = 'Train' if train else 'Test'
    model.eval()
    with torch.no_grad():
        with h5py.File(args.h5_file, 'r') as h5_file:
            for video in tqdm(video_list, desc=f'{status} video'):
                features = []

                video_class, video_info = os.path.split(video.strip())
                video_name, seg_num = video_info.rsplit('_', maxsplit=1)
                idx_all = list(np.arange(int(seg_num)))
                keys = list(
                    map(lambda idx: f'{os.path.splitext(os.path.basename(video_name))[0]}_{idx:06d}',
                        idx_all))

                for key in keys:
                    raw_frames = h5_file[key][:]
                    frames = decode_frames(raw_frames, args.smaller_size)

                    inputs = transforms(frames)  # t, h, w, c
                    inputs = torch.from_numpy(inputs.transpose([3, 0, 1, 2])).unsqueeze(0)  # b, c, t, h, w
                    inputs = inputs.cuda()

                    outputs = model.extract_features(inputs)
                    features.append(outputs.squeeze(0).permute(1, 2, 3, 0).cpu().numpy())

                save_path = os.path.join(args.save_dir, status, video_class)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(os.path.join(save_path, video_name + '.npy'),
                        np.concatenate(features, axis=0).squeeze((1, 2)))


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
