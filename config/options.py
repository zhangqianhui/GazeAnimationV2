import argparse
import os
from PyLib.utils import makefolders
from abc import abstractmethod

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):

        parser.add_argument('--data_dir', type=str,
                            default='../dataset/NewGazeData', help='path to images')
        parser.add_argument('--pretrain_path', type=str,
                            default='./sg_pre_model_g/', help='pretrained model path')
        parser.add_argument('--inception_path', type=str, default='../pretrained/')
        parser.add_argument('--img_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--nef', type=int, default=32, help='# of encoder filters in frist decov layer')
        parser.add_argument('--ngf', type=int, default=16, help='# of generator filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=16, help='# of discriminator filters in first conv layer')
        parser.add_argument('--n_layers_e', type=int, default=3, help='layers of sae')
        parser.add_argument('--n_layers_g', type=int, default=5, help='layers of generator')
        parser.add_argument('--n_layers_d', type=int, default=5, help='layers of d model')
        parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--exper_name', type=str, default='log3_25_1', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='./logs', help='logs for tensorboard')
        parser.add_argument('--sample_dir', type=str, default='./sample_dir', help='dir for sample images')
        parser.add_argument('--test_sample_dir', type=str, default='test_sample_dir', help='test sample images are saved here')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):

        opt.checkpoints_dir = os.path.join(opt.exper_name, opt.checkpoints_dir)
        opt.sample_dir = os.path.join(opt.exper_name, opt.sample_dir)

        opt.test_sample_dir = os.path.join(opt.exper_name, opt.test_sample_dir)

        opt.log_dir = os.path.join(opt.exper_name, opt.log_dir)
        makefolders([opt.inception_path, opt.checkpoints_dir, opt.sample_dir, opt.test_sample_dir, opt.log_dir])

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        # save to the disk
        if opt.isTrain:
            file_name = os.path.join(opt.checkpoints_dir, 'opt.txt')
        else:
            file_name = os.path.join(opt.checkpoints_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    @abstractmethod
    def parse(self):
        pass
