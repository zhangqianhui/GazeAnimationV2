from .options import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self, parser):

        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.isTrain = False
        return parser

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        self.opt = opt

        return self.opt