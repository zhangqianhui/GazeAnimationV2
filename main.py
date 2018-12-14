import os
from config import Config
from Dataset import Dataset
from Inpainting_GAN import Inpainting_GAN

config = Config()
os.environ['CUDA_VISIBLE_DEVICES']= str(config.gpu_id)

if __name__ == "__main__":

    d_ob = Dataset(config)
    igan = Inpainting_GAN(d_ob, config)
    igan.build_model()
    if config.is_training:
        igan.train()
    else:
        igan.test2()


