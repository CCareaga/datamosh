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

from mosh_utils import np_to_torch, read_frames, get_temp_name, write_frames

sys.path.append('RAFT/core')
sys.path.append('RAFT')

from raft import RAFT
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def datamosh(args):
    """Method to create datamosh effect on input videos
    """

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

    # if they want to mosh multiple videos get the sorted files
    if args.input_folder is not None:
        print("Read video files from {}".format(args.input_folder))
        vid_paths = sorted(glob.glob(f"{args.input_folder}/*"))
    else:
        print("Processing the specified video files")
        vid_paths = [args.path_1, args.path_2]

    # if we are in reverse mode, the order of all the videos should be reversed
    print("Datamoshing in reverse={}".format(args.reverse))
    if args.reverse:
        vid_paths = vid_paths[::-1]

    # load up the first video in the mosh
    print("Reading video: {}".format(vid_paths[0]))
    init_vid = read_frames(vid_paths[0], h=h, w=w)

    # write this video out to temp storage, will remove later
    # but we need it to have the same format as our other output vids
    print("Writing initial video: {}".format(get_temp_name(vid_paths[0])))
    write_frames(init_vid, get_temp_name(vid_paths[0]), height=h, width=w)

    # if we are in reverse mode, we also reverse the video frames
    if args.reverse:
        init_vid = init_vid[::-1]

    # save this frame for warping later
    start_frame = init_vid[-1]
    del init_vid

    # looping through all the videos
    for vid, vid_path in enumerate(vid_paths[1:]):
        print("Processing video: {}/{}".format(vid + 1, len(vid_paths[1:])))

        flows = []
        outputs = []
        warps = []
        masks = []

        curr_vid = read_frames(vid_path, h=h, w=w)

        # if in reverse mode always reverse the frames
        if args.reverse:
            curr_vid = curr_vid[::-1]

        for i, (image2, image1) in tqdm(enumerate(zip(curr_vid[:-1], curr_vid[1:])), total=len(curr_vid)):

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

        curr_vid = [np.array(f).astype(np.uint8) for f in curr_vid]
        # start_frame = cv2.GaussianBlur(start_frame, (7, 7), 3)
        warped = torch.from_numpy(start_frame).permute(2, 0, 1).unsqueeze(0).float()
        fg_mask = torch.ones_like(warped) * 255.

        print("Creating datamosh...")
        for flw in flows:
            grid = create_meshgrid(h, w, False)
            grid += flw.permute(0, 2, 3, 1).cpu()

            warped = remap(warped, grid[..., 0], grid[..., 1], mode='nearest', align_corners=True)
            fg_mask = remap(fg_mask, grid[..., 0], grid[..., 1], mode='nearest', align_corners=True)

            # flw = flow.squeeze(0)
            # flow_mag = torch.sqrt(flw[0, ...] ** 2 + flw[1, ...] ** 2)            
            # mag_mask = (flow_mag < np.percentile(flow_mag.numpy(), args.flow_perc)).int()
            # fg_mask *= mag_mask

            masks.append(fg_mask.squeeze(0).permute(1, 2, 0).numpy())
            warps.append(warped.squeeze(0).permute(1, 2, 0).numpy())

        for orig, warped, mask in zip(curr_vid, warps, masks):
            mask = mask.astype(bool)
            warped[~mask] = orig[~mask]

            outputs.append(warped)

        del masks
        del warps
        del flows
        gc.collect()

        start_frame = outputs[-1]

        # if we are in reverse mode, unreverse the clip to play forward
        if args.reverse:
            outputs = outputs[::-1]

        write_frames(outputs, get_temp_name(vid_path), height=h, width=w)

    # if reverse mode, we already reversed these paths so undo that
    if args.reverse:
        vid_paths = vid_paths[::-1]

    temp_names = [get_temp_name(x) for x in vid_paths]

    fnames_file = os.path.join(args.input_folder, 'files.txt')

    # write out a file for ffmpeg to concatenate these clips.
    # the first clip (or the last clip in reverse mode) should not be moshed
    # so it never gets written out to a temporary file
    with open(fnames_file, 'w+') as f:
        for fname in temp_names:
            f.write(f"file '{basename(fname)}'\n")

    call(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', fnames_file, '-c', 'copy', args.out_path])

    for tmp_name in temp_names:
        os.remove(tmp_name)

    os.remove(fnames_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('-p1', '--path_1', help="path to first video")
    parser.add_argument('-p2', '--path_2', help="path to second video")
    parser.add_argument('-if', '--input_folder', help="folder of input videos (ordered by name)", default=None)
    parser.add_argument('-op', '--out_path', help="path to output video, e.g. ./output.mp4")

    # flow arguments
    parser.add_argument('-rv', '--reverse', action='store_true', help='flag to reverse the output')
    parser.add_argument('-rt', '--raft', action='store_true', help='flag to use raft flow')
    parser.add_argument('-ri', '--raft_iter', default=5, type=int, help='raft iterations')
    parser.add_argument('-fs', '--flow_speed', type=float, help='optical flow speed', default=1.0)
    parser.add_argument('-fp', '--flow_perc', type=float, help='threshold on magnitude before dropping pixels',
                        default=0.0)

    # image arguments
    parser.add_argument('-fh', '--height', help='frame height', default=720, type=int)
    parser.add_argument('-fw', '--width', help='frame height', default=1080, type=int)
    args = parser.parse_args()

    datamosh(args)
