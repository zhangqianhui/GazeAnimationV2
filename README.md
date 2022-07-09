## GazeAnimationV2 - Official Tensorflow Implementation (TIP)

> ***Unsupervised High-Resolution Portrait Gaze Correction and Animation*** <br>
> Paper: https://arxiv.org/abs/2207.00256 <br>

## Dependencies

```bash
Python=3.6
pip install -r requirements.txt

```
Or Using Conda

```bash
-conda create -name GazeA python=3.6
-conda install tensorflow-gpu=1.9 or higher
```
Other packages installed by pip.

## Usage

- Clone this repo:
```bash
git clone https://github.com/zhangqianhui/GazeAnimation.git
cd GazeAnimation
git checkout GazeAnimationV2
```


- Download the CelebAGaze dataset

  Download the tar of CelebGaze dataset from [Google Driver Linking](https://drive.google.com/file/d/1_6f3wT72mQpu5S2K_iTkfkiXeeBcD3wn/view?usp=sharing).
  
  ```bash
  cd your_path
  tar -xvf CelebAGaze.tar
  ```
  
- Download the CelebHQGaze dataset

Download the tar of CelebHQGaze dataset from [Google Driver Linking](https://drive.google.com/file/d/1Q7iOc3ZWnyld2fbw6yYR0EIkF0m0FERN/view?usp=sharing).

```bash
cd your_path
tar -xvf CelebHQGaze.tar
```


## Experiment Result 

### Gaze Correction on CelebHQGaze

<p align="center"><img width="100%" src="img/correction.jpg" /></p>

```
@article{zhang2022unsupervised,
  title={Unsupervised High-Resolution Portrait Gaze Correction and Animation},
  author={Zhang, Jichao and Chen, Jingjing and Tang, Hao and Sangineto, Enver and Wu, Peng and Yan, Yan and Sebe, Nicu and Wang, Wei},
  journal={IEEE Transactions on Image Processing},
  year={2022}
}
```
