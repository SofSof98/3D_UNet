import argparse, os

class BaseOptions():
    """This class defines options for training, testing and prediction"""

    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser):
        self.initialized = True
        parser.add_argument('--check_consistency', action='store_true', required=False, help='Check consistency of data by comparing NRRD headders.')
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        msg = ''
        msg += '--------------- Options -----------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            msg += '{:>20}: {:<30}{}\n'.format(str(k), str(v), comment)
        msg += '----------------- End -------------------'
        print(msg)

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        self.print_options(opt)
        return self.opt