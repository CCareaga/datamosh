#!/usr/bin/env python
# coding: utf-8

import glob
import os
import sys
import gc

from subprocess import call
from ntpath import basename

sys.path.append('RAFT/core')
sys.path.append('RAFT')

from raft import RAFT
from utils.utils import InputPadder

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import cv2

import kornia
from kornia.utils import create_meshgrid

from tqdm import tqdm
import argparse
from argparse import Namespace

parser = argparse.ArgumentParser()
parser.add_argument('--path1', help="path to first video")
parser.add_argument('--path2', help="path to second video")
parser.add_argument('--input_folder', help="folder of input videos (ordered by name)", default='')

parser.add_argument('--out_path', help="path to output video")
parser.add_argument('--reverse', action='store_true', help='flag to reverse the output')
parser.add_argument('--raft', action='store_true', help='flag to use raft flow')
parser.add_argument('--raft_iter', type=int, help='raft iterations')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def convert_frame(f):
    f = np.array(f).astype(np.uint8)
    return torch.from_numpy(f).permute(2, 0, 1).float().unsqueeze(0)

def get_temp_name(f):
    base = os.path.splitext(f)[0]
    return f'{base}_tmp.avi'

def read_frames(fname, h=720, w=1280):

    cap = cv2.VideoCapture(fname)

    frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h))

        frames.append(frame)

    cap.release()
    
    return frames

def write_frames(frames, fname):

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(fname, fourcc, 29.97, (w, h))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame.astype(np.uint8))

    out.release()

if args.raft:
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

# TODO: make this a command line arg
h, w = 720, 1080

# if they want to mosh multiple videos get the sorted files
if args.input_folder != '':
    vid_paths = sorted(glob.glob(f"{args.input_folder}/*"))
else:
    vid_paths = [args.path1, args.path2]

# if we are in reverse mode, the order of all the videos should be reversed
if args.reverse:
    vid_paths = vid_paths[::-1]

# load up the first video in the mosh
init_vid = read_frames(vid_paths[0], h=h, w=w)

# write this video out to temp storage, will remove later
# but we need it to have the same format as our other output vids
write_frames(init_vid, get_temp_name(vid_paths[0]))

# if we are in reverse mode, we also reverse the video frames
if args.reverse:
    init_vid = init_vid[::-1]

# save this frame for warping later
start_frame = init_vid[-1]
del init_vid

for vid_path in vid_paths[1:]:
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
                image1 = convert_frame(image1).to(DEVICE)
                image2 = convert_frame(image2).to(DEVICE)
    
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
    
                flow_low, flow_up = model(image1, image2, iters=args.raft_iter, test_mode=True)
    
                flow = flow_up.cpu()
        
        else:
            image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
            flow = torch.from_numpy(np.array(flow)).permute(2, 0, 1).unsqueeze(0)
        
        flows.append(flow)
    
    curr_vid = [np.array(f).astype(np.uint8) for f in curr_vid]

    warped = torch.from_numpy(start_frame).permute(2, 0, 1).unsqueeze(0).float()
    fg_mask = torch.ones_like(warped) * 255.
    
    for flw in flows:
    
        grid = create_meshgrid(h, w, False)
        grid += flw.permute(0, 2, 3, 1).cpu()
    
        warped = kornia.geometry.transform.remap(warped, grid[..., 0], grid[..., 1], mode='nearest', align_corners=True)
        fg_mask = kornia.geometry.transform.remap(fg_mask, grid[..., 0], grid[..., 1], mode='nearest', align_corners=True)
    
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

    write_frames(outputs, get_temp_name(vid_path))

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
