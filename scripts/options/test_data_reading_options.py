from .base_options import BaseOptions

class ReadOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--shuffle', action='store_false', required=False, help='Shall the data be shuffled?')
        parser.add_argument('--no_data_info', action='store_false', required=False, help='Print information about the data (how the split into train, test and val data looks like).')
        parser.add_argument('--return_ratio', action='store_true', required=False, help='Return the ratio of your data, to see, if it is unbalanced.')
        parser.add_argument('--csv_file', type=str, default='True', required=True, help='Path to .csv file that contains information about data.')
        parser.add_argument('--data_root', type=str, default='True', required=True, help='Flip data for data augmentation.')
        parser.add_argument('--results_path', type=str, default='results/test_data_reading/', required=False, help='Flip data for data augmentation.')
        return parser