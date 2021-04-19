import os

import cv2
import numpy as np
import torch


def np_to_torch(frame: np.ndarray) -> torch.Tensor:
    """Method to convert numpy array to torch
    """
    frame = np.array(frame).astype(np.uint8)
    return torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)


def get_temp_name(path: str) -> str:
    """Method to get temporary file name
    """
    base = os.path.splitext(path)[0]
    return f'{base}_tmp.avi'


def read_frames(fname: str, h: int = 720, w: int = 1080) -> list:
    """Method to read frames from a video path
    """
    cap = cv2.VideoCapture(fname)

    frames = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h))

        frames.append(frame)

    cap.release()

    return frames


def write_frames(frames: list, fname: str, height: int, width: int):
    """Method to write frames to a video
    """

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(fname, fourcc, 29.97, (width, height))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame.astype(np.uint8))

    out.release()


def check_directory(path: str):
    """Method to create directory if it does not exist
    """

    if not os.path.exists(path):
        os.makedirs(path)
