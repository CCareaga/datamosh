#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import sys

sys.path.append('RAFT/core')
sys.path.append('RAFT')
sys.path.append('pyflow')

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

DEVICE = 'cuda'

def convert_frame(f):
    f = np.array(f).astype(np.uint8)
    return torch.from_numpy(f).permute(2, 0, 1).float().unsqueeze(0)

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

if args.raft:
    dummy = Namespace(
        small=True, 
        alternate_corr=False, 
        model='RAFT/models/raft-small.pth',
        # path='RAFT/demo-frames/',
        mixed_precision=True
    )

    model = torch.nn.DataParallel(RAFT(dummy))
    model.load_state_dict(torch.load(dummy.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

h, w = 720, 1080

if args.input_folder != '':
    vid_paths = sorted(glob.glob(f"{args.input_folder}/*"))
else:
    vid_paths = [args.path1, args.path2]

if args.reverse:
    vid_paths = vid_paths[::-1]

init_vid = read_frames(vid_paths[0], h=h, w=w)

if args.reverse:
    init_vid = init_vid[::-1]

all_frames = init_vid

for vid_path in vid_paths[1:]:
    curr_vid = read_frames(vid_path, h=h, w=w)
    
    if args.reverse:
        curr_vid = curr_vid[::-1]
    
    flows = []
    
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
    
    start_frame = all_frames[-1]
    warped = torch.from_numpy(start_frame).permute(2, 0, 1).unsqueeze(0).float()
    fg_mask = torch.ones_like(warped) * 255.
    
    warps = []
    masks = []
    
    for flw in flows:
    
        grid = create_meshgrid(h, w, False)
        grid += flw.permute(0, 2, 3, 1).cpu()
    
        warped = kornia.geometry.transform.remap(warped, grid[..., 0], grid[..., 1], mode='nearest', align_corners=True)
        fg_mask = kornia.geometry.transform.remap(fg_mask, grid[..., 0], grid[..., 1], mode='nearest', align_corners=True)
    
        masks.append(fg_mask.squeeze(0).permute(1, 2, 0).numpy())
        warps.append(warped.squeeze(0).permute(1, 2, 0).numpy())
    
    outputs = []

    for orig, warped, mask in zip(curr_vid, warps, masks):
        mask = mask.astype(bool)
        warped[~mask] = orig[~mask]
        outputs.append(warped)
    
    all_frames += outputs

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(args.out_path, fourcc, 30.0, (w, h))

if args.reverse:
    all_frames = all_frames[::-1]

for frame in all_frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame.astype(np.uint8))

out.release()

