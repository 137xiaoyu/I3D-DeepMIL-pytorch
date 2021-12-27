import cv2
import os

if __name__ == '__main__':
    input_video = 'D:/Users/Chase/Desktop/Abuse001_x264.mp4'
    output_dir = 'D:/Users/Chase/Desktop/Abuse001_x264/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        print('Capture is not opened!')

    print(f'fps={cap.get(cv2.CAP_PROP_FPS)}\nnum_frames={cap.get(cv2.CAP_PROP_FRAME_COUNT)}')

    t = 0
    while True:
        t += 1
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f'{t:05d}.jpg'), frame)
        else:
            break
