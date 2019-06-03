# GazeCorrection:Self-Guided Eye Manipulation in the wild using Self-Supervised Generative Adversarial Networks
The code of paper [GazeCorrection:Self-Guided Eye Manipulation in the wild using Self-Supervised Generative Adversarial Networks](https://arxiv.org/). 

--------------------------------------------

### [Project page](https://nips2019.wixsite.com/gazegan) | [Paper](https://arxiv.org/) | 

![](img/t_input2.jpg)
![](img/t_output2.jpg)

![](img/gif/gif1/age.gif)
![](img/gif/gif2/age.gif)
![](img/gif/gif3/age.gif)

# Abstract
Gaze correction aims to redirect the person's gaze into the camera by manipulating the eye region, and
it can be considered as a specific image resynthesis problem. Gaze correction has a wide
range of applications in real life, such as taking a picture with staring at the camera.
In this paper, we propose a novel method that is based on the inpainting model to learn from
the face image to fill in the missing eye regions with new contents representing corrected eye gaze.
Moreover, our model does not require the training dataset labeled with
the specific head pose and eye angle information, thus, the training data is easy to collect.
To retain the identity information of the eye region in the original input,
we propose a self-guided pretrained model to learn the angle-invariance feature.
Experiments show our model achieves very compelling gaze-corrected results in the wild dataset which is collected
from the website and will be introduced in details. 

## Network Architecture

![](img/model.jpg)


## 

## Dependencies
* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [Tensorflow 1.4+](https://github.com/tensorflow/tensorflow)


## Usage

- Clone this repo:
```bash
git clone https://github.com/zhangqianhui/Eye_Rotation_GAN.git
```
- Download the Eye_Rotation dataset

Coming soon!!!

- Train the model using the default parameter
```bash
python main.py 
```


# The process of Experiments

![](img/output.jpg)

# Reference code

- [Sparsely_Grouped_GAN](https://github.com/zhangqianhui/Sparsely-Grouped-GAN)

- [DCGAN tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

- [Spectral Norm tensorflow](https://github.com/taki0112/Spectral_Normalization-Tensorflow)

- [Exemplar-GAN-Eye-Inpainting-Tensorflow](https://github.com/zhangqianhui/Exemplar-GAN-Eye-Inpainting-Tensorflow.git)

- [ELEGANT](https://github.com/Prinsphield/ELEGANT)
