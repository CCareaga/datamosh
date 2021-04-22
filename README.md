# Datamosh

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

## Demo
Use the `demo.sh` to create datamosh using the example videos in `videos`
```python
./datamosh_demo.sh
```