# Deep Patch Visual Odometry
This repository contains the source code for our papers:

[Deep Patch Visual Odometry](https://arxiv.org/pdf/2208.04726.pdf)<br/>
Zachary Teed<sup>\*</sup>, Lahav Lipson<sup>\*</sup>, Jia Deng <sub></sub><br/>
[Deep Patch Visual SLAM](http://arxiv.org/pdf/2408.01654)<br/>
Lahav Lipson, Zachary Teed, Jia Deng<br/>
<a target="_blank" href="https://colab.research.google.com/drive/1VSFGNB7YCveqKF7XNz4RlV9EnfQA3fhQ?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a><a target="_blank" href="https://github.com/princeton-vl/DPVO_Docker">
  <img src="https://img.shields.io/badge/Docker-grey?logo=Docker" alt="Open In Colab"/>
</a>

[<img src="https://i.imgur.com/6ZQPbR1.png?1" width="600">](https://www.youtube.com/watch?v=e5wanf71YFs)

```
@article{teed2023deep,
   title={Deep Patch Visual Odometry},
   author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
   journal={Advances in Neural Information Processing Systems},
   year={2023}
 }
```
```
@inproceedings{lipson2024deep,
    author={Lipson, Lahav and Teed, Zachary and Deng, Jia},
    title={{Deep Patch Visual SLAM}},
    booktitle={European Conference on Computer Vision},
    year={2024}
}
```
## Setup and Installation
The code was tested on Ubuntu 20/22 and Cuda 11/12.</br>

Clone the repo
```
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO
```
Create and activate the dpvo anaconda environment
```
conda env create -f environment.yml
conda activate dpvo
```

Next install the DPVO package
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install DPVO
pip install .

# download models and data (~2GB)
./download_models_and_data.sh
```


### Recommended - Install the Pangolin Viewer
Note: You will need to have CUDA 11 and CuDNN installed on your system.

1. Step 1: Install Pangolin (need the custom version included with the repo)
```
./Pangolin/scripts/install_prerequisites.sh recommended
mkdir Pangolin/build && cd Pangolin/build
cmake ..
make -j8
sudo make install
cd ../..
```

2. Step 2: Install the viewer
```bash
pip install ./DPViewer
```

For installation issues, our [Docker Image](https://github.com/princeton-vl/DPVO_Docker) supports the visualizer.

### Classical Backend (optional)

We provide a classical backend for closing very large loops, which requires extra installation.

Step 1. Install the OpenCV C++ API. On Ubuntu, you can use
```bash
sudo apt-get install -y libopencv-dev
```
Step 2. Install DBoW2
```bash
cd DBoW2
mkdir -p build && cd build
cmake .. # tested with cmake 3.22.1 and gcc/cc 11.4.0 on Ubuntu
make # tested with GNU Make 4.3
sudo make install
cd ../..
```

Step 3. Install the image retrieval
```bash
pip install ./DPRetrieval
```

## Demos
DPVO can be run on any video or image directory with a single command. Note you will need to have installed DPViewer to visualize the reconstructions in real-time. You can also save the completed reconstructions and view them in COLMAP. The pretrained models can be downloaded from google drive [models.zip](https://drive.google.com/file/d/1dRqftpImtHbbIPNBIseCv9EvrlHEnjhX/view?usp=sharing) if you have not already run the download script. 


```bash
python demo.py \
    --imagedir=<path to image directory or video> \
    --calib=<path to calibration file> \
    --viz # enable visualization
    --plot # save trajectory plot
    --save_ply # save point cloud as a .ply file
    --save_trajectory # save the predicted trajectory as .txt in TUM format
    --save_colmap # save point cloud + trajectory in the standard COLMAP text format
```

### iPhone
```bash
python demo.py --imagedir=movies/IMG_0492.MOV --calib=calib/iphone.txt --stride=5 --plot --viz
```

### TartanAir
Download a sequence from [TartanAir](https://theairlab.org/tartanair-dataset/) (several samples are availabe from download directly from the webpage)
```bash
python demo.py --imagedir=<path to image_left> --calib=calib/tartan.txt --stride=1 --plot --viz
```

### EuRoC
Download a sequence from [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) (download ASL format)
```bash
python demo.py --imagedir=<path to mav0/cam0/data/> --calib=calib/euroc.txt --stride=2 --plot --viz
```

## SLAM Backends
To run DPVO with a SLAM backend (i.e., DPV-SLAM), add
```bash
--opts LOOP_CLOSURE True
```
to any `evaluate_X.py` script or to `demo.py`

If installed, the classical backend can also be enabled using 
```
--opts CLASSIC_LOOP_CLOSURE True
```

## Evaluation
We provide evaluation scripts for TartanAir, EuRoC, TUM-RGBD and ICL-NUIM. Up to date result logs on these datasets can be found in the `logs` directory.

### TartanAir:
Results on the validation split and test set can be obtained with the command:
```
python evaluate_tartan.py --trials=5 --split=validation --plot --save_trajectory
```

### EuRoC:
```
python evaluate_euroc.py --trials=5 --plot --save_trajectory
```

### TUM-RGBD:
```
python evaluate_tum.py --trials=5 --plot --save_trajectory
```

### ICL-NUIM:
```
python evaluate_icl_nuim.py --trials=5 --plot --save_trajectory
```

### KITTI:
```
python evaluate_kitti.py --trials=5 --plot --save_trajectory
```

## Training
Make sure you have run `./download_models_and_data.sh`. Your directory structure should look as follows

```Shell
├── datasets
    ├── TartanAir.pickle
    ├── TartanAir
        ├── abandonedfactory
        ├── abandonedfactory_night
        ├── ...
        ├── westerndesert
    ...
```

To train (log files will be written to `runs/<your name>`). Model will be run on the validation split every 10k iterations
```
python train.py --steps=240000 --lr=0.00008 --name=<your name>
```

## Change Log
* **Aug 2022**: Initial release
* **Sep 2022**: Add link to docker
* **Mar 2023**: Google Colab, TUM + ICL-NUIM evaluation code, flags for saving output
* **July 2024**: Add DPV-SLAM. Update output-saving utilities.


## Acknowledgements
* Our Viewer is adapted from DSO.
