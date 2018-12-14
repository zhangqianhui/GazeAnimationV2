import os

from Dataset2 import Dataset
from Dataset2 import mkdir
from config import Config
from Inpainting_pg_gan import Inpainting_GAN

config = Config()
os.environ['CUDA_VISIBLE_DEVICES']= str(config.gpu_id)

if __name__ == "__main__":

    d_ob = Dataset(config)
    pggan_base_model_write = config.write_model_path()
    pggan_base_sample_path = config.sample_path()

    flag = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    read_flag = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]

    for i in range(len(flag)):

        # not the 4x4 for starting but 16x16 for starting
        i = i + 11

        is_trans = False if (i % 2 == 0) else True
        pggan_model_write = pggan_base_model_write + "/{}/".format(flag[i])
        pggan_model_read = pggan_base_model_write + "/{}/".format(read_flag[i])
        pggan_sample_path = pggan_base_sample_path + "/{}_{}/".format(flag[i], i % 2)

        mkdir(pggan_model_write)
        mkdir(pggan_sample_path)

        igan = Inpainting_GAN(d_ob, config, model_write=pggan_model_write
                              , model_read=pggan_model_read, sample_path=pggan_sample_path,
                              pg=flag[i], is_trans=is_trans, base_step=i)
        igan.build_model()

        if config.is_training:
            igan.train()

        else:
            igan.test2()


