![Alt Text](examples/logo.png)

This repository contains the code for applying datamosh or 'glitchy' visual effect in videos.

Example datamoshed videos with different video shots -- panning, zooming, looping, etc: [YouTube Playlist](https://youtube.com/playlist?list=PLxQH-axrX98g7myRfhSe2XWomf-mUzuea)

## Setup
Clone the datamosh repository and then recursively clone the optical flow [RAFT](https://github.com/princeton-vl/RAFT) dependency.
```python
git clone https://www.github.com/CCareaga/datamosh.git
git clone --recurse-submodules -j8 https://github.com/princeton-vl/RAFT.git

conda create -n datamosh
conda activate datamosh
```

Follow [RAFT](https://github.com/princeton-vl/RAFT) guidelines to install the dependencies
```python
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
```
Additional python dependencies for datamosh
```python
pip install kornia tqdm
```

## Demo
1. Animating Looping

![Alt Text](examples/looping.gif)

```python
python loop_mosh.py --input_path ./examples/video_01.MOV --output_path ./examples/looping.gif --gif
```

2. Panning

![Alt Text](examples/panning.gif)

3. Zooming

![Alt Text](examples/zooming.gif)