
import argparse
import gc
import glob
import os
import sys
from argparse import Namespace
from ntpath import basename
from subprocess import call

import cv2
import numpy as np
import torch
from kornia.geometry.transform import remap
from kornia.utils import create_meshgrid
from tqdm import tqdm

from mosh_utils import np_to_torch, read_frames, get_temp_name, write_frames, write_gif

sys.path.append('RAFT/core')
sys.path.append('RAFT')

from raft import RAFT
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def loop_mosh(args):
    # if using CNN-based optical flow -- RAFT
    if args.raft:
        print("Datamoshing using RAFT...")
        dummy = Namespace(
            small=True,
            alternate_corr=False,
            model='RAFT/models/raft-small.pth',
            # path='RAFT/demo-frames/',
            mixed_precision=True
        )

        model = torch.nn.DataParallel(RAFT(dummy))
        model.load_state_dict(torch.load(dummy.model, map_location=DEVICE))

        model = model.module
        model.to(DEVICE)
        model.eval()

    # frame height and width
    h, w = args.height, args.width

    # load up the video to be moshed
    print("Reading video: {}".format(args.input_path))
    vid_frames = read_frames(args.input_path, h=h, w=w)
    vid_frames = vid_frames[::-1]

    flows = []
    outputs = []
    warps = []
    masks = []

    for i, (image2, image1) in tqdm(enumerate(zip(vid_frames[:-1], vid_frames[1:])), total=len(vid_frames)):

        if args.raft:
            with torch.no_grad():
                image1 = np_to_torch(image1).to(DEVICE)
                image2 = np_to_torch(image2).to(DEVICE)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=args.raft_iter, test_mode=True)
                flow = flow_up.cpu()

        else:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

            # TODO: pass the optical flow parameters through argparse
            flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = torch.from_numpy(np.array(flow)).permute(2, 0, 1).unsqueeze(0)

        flows.append(flow * args.flow_speed)

    vid_frames = [np.array(f).astype(np.uint8) for f in vid_frames]
   
    # the frames are reversed so this is actually the first frame
    start_frame = vid_frames[-1]

    warped = torch.from_numpy(start_frame).permute(2, 0, 1).unsqueeze(0).float()
    fg_mask = torch.ones_like(warped) * 255.

    print("Creating datamosh...")
    for flw in flows:
        grid = create_meshgrid(h, w, False)
        grid += flw.permute(0, 2, 3, 1).cpu()

        warped = remap(warped, grid[..., 0], grid[..., 1], mode='nearest', align_corners=True)
        fg_mask = remap(fg_mask, grid[..., 0], grid[..., 1], mode='nearest', align_corners=True)

        masks.append(fg_mask.squeeze(0).permute(1, 2, 0).numpy())
        warps.append(warped.squeeze(0).permute(1, 2, 0).numpy())

    for orig, warped, mask in zip(vid_frames, warps, masks):
        mask = mask.astype(bool)
        warped[~mask] = orig[~mask]

        outputs.append(warped)

    del masks
    del warps
    del flows
    gc.collect()
    
    outputs = outputs[::-1]
    write_gif(outputs, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('-ip', '--input_path', help="path to the input video, e.g. ./input.mp4", default=None)
    parser.add_argument('-op', '--output_path', help="path to output video, e.g. ./output.mp4")

    # flow arguments
    parser.add_argument('-rt', '--raft', action='store_true', help='flag to use raft flow')
    parser.add_argument('-ri', '--raft_iter', default=5, type=int, help='raft iterations')
    parser.add_argument('-fs', '--flow_speed', type=float, help='optical flow speed', default=1.0)

    # image arguments
    parser.add_argument('-fh', '--height', help='frame height', default=720, type=int)
    parser.add_argument('-fw', '--width', help='frame height', default=1080, type=int)
    args = parser.parse_args()

    loop_mosh(args)
