![Alt Text](examples/logo.png)

This repository contains the code for applying datamosh or 'glitchy' visual effect in videos.

Example datamoshed videos with different video shots -- panning, zooming, looping, etc: [YouTube Playlist](https://youtube.com/playlist?list=PLxQH-axrX98g7myRfhSe2XWomf-mUzuea)

## Setup
Clone the datamosh repository and then recursively clone the optical flow [RAFT](https://github.com/princeton-vl/RAFT) dependency.
```
git clone https://www.github.com/CCareaga/datamosh.git
git clone --recurse-submodules -j8 https://github.com/princeton-vl/RAFT.git

conda create -n datamosh
conda activate datamosh
```

Follow [RAFT](https://github.com/princeton-vl/RAFT) guidelines to install the dependencies
```
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```
Additional python dependencies for datamosh
```
pip install kornia tqdm
```

## Demo
1. Animating Looping

![Alt Text](examples/looping.gif)

```
python loop_mosh.py --input_path ./examples/video_01.MOV --output_path ./examples/looping.gif --gif
```

----
2. Panning

For videos with panning or zooming shots, use the `mosh.py` with either a pair of videos or a directory of videos. Refer to the following command line arguments to run the script. Additionally, you can choose between different optical flow frameworks. The default framework is Farneback, to use RAFT, set `--raft`. The default datamosh effect is in forward direction, to create the reverse effect use `--reverse`. 
```
optional arguments:
  -h, --help            show this help message and exit
  -p1 PATH_1, --path_1 PATH_1
                        path to first video
  -p2 PATH_2, --path_2 PATH_2
                        path to second video
  -if INPUT_FOLDER, --input_folder INPUT_FOLDER
                        folder of input videos (ordered by name)
  -op OUT_PATH, --out_path OUT_PATH
                        path to output video, e.g. ./output.mp4
  -rv, --reverse        flag to reverse the output
  -rt, --raft           flag to use raft flow
  -ri RAFT_ITER, --raft_iter RAFT_ITER
                        raft iterations
  -fs FLOW_SPEED, --flow_speed FLOW_SPEED
                        optical flow speed
  -fp FLOW_PERC, --flow_perc FLOW_PERC
                        threshold on magnitude before dropping pixels
  -fh HEIGHT, --height HEIGHT
                        frame height
  -fw WIDTH, --width WIDTH
                        frame height

```
![Alt Text](examples/panning.gif)
```python
python mosh.py --input_path ./<video_directory>/ --output_path ./<output.mp4>
```

3. Zooming

![Alt Text](examples/zooming.gif)