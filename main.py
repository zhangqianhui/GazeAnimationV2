import os
from Dataset import Dataset
from config import Config
from GazeGAN import Gaze_GAN

if __name__ == "__main__":

    config = Config()
    print config.exp_name
    if config.CUDA:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

    dataset = Dataset(config)
    gaze_gan = Gaze_GAN(dataset, config)
    gaze_gan._init_inception()
    gaze_gan.build_model()

    if config.is_training:
        gaze_gan.train()
    else:
        gaze_gan.test()