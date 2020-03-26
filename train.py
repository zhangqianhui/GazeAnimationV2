from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from Dataset import Dataset
from GazeGAN import Gaze_GAN
from config.train_options import TrainOptions

print("1")
opt = TrainOptions().parse()
print("2")

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

if __name__ == "__main__":

    dataset = Dataset(opt)
    gaze_gan = Gaze_GAN(dataset, opt)
    gaze_gan.build_model()
    gaze_gan.train()