from .options import BaseOptions

class TrainOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_model_freq', type=int, default=20000, help='frequency of saving checkpoints')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--capacity', type=int, default=5000, help='capacity for queue in training')
        parser.add_argument('--num_threads', type=int, default=10, help='thread for reading data in training')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=100000, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=50000, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_d', type=float, default=0.0001, help='initial learning rate for Adam in d')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='initial learning rate for Adam in g')
        parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
        parser.add_argument('--loss_type', type=str, default='hinge', choices=['hinge', 'vgan', 'wgan'], help='using type of gan loss')
        parser.add_argument('--lam_gp', type=float, default=10.0, help='weight for gradient penalty of gan')
        parser.add_argument('--lam_p', type=float, default=1.0, help='perception loss in g')
        parser.add_argument('--lam_r', type=float, default=1.0, help='weight for recon loss in g')
        parser.add_argument('--lam_ss', type=float, default=1, help='self-supervised loss in g')
        parser.add_argument('--use_sp', action='store_true', help='use spetral normalization')
        parser.add_argument('--is_ss', action='store_true', help='whether using self supervised')
        parser.add_argument('--pos_number', type=int, default=4, help='position')
        parser.add_argument('--crop_w', type=int, default=50, help='the size of cropped eye region')
        parser.add_argument('--crop_h', type=int, default=30, help='the size of crooped eye region')

        self.isTrain = True
        return parser

    def parse(self):

        opt = self.gather_options()
        print("opt")
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt
