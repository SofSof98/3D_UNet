from .base_options import BaseOptions
import sys

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--shuffle', action='store_false', required=False, help='Shall the data be shuffled?')
        parser.add_argument('--no_data_info', action='store_true', required=False, help='Print information about the data (how the split into train, test and val data looks like).')
        parser.add_argument('--return_ratio', action='store_true', required=False, help='Return the ratio of your data.')
        parser.add_argument('--trained_model', type=str, default='trained_model.sausage', required=False, help='Name for the trained model.')
        parser.add_argument('--results_path', type=str, default='results/train/', required=False, help='Path to store the results')
        parser.add_argument('--csv_file', type=str, default='True', required=True, help='Path to .csv file that contains information about data.')
        parser.add_argument('--data_root', type=str, default='True', required=True, help='Path to data.')
        parser.add_argument('--loss_fn', type=str, default='cross_entropy', required=False, help='Choose loss function: {"weighted_cross_entropy", "cross_entropy", "dice"}.')
        parser.add_argument('--weights', type=float, default=None, nargs='*', required=False, help='Choose weights for weighted cross entropy loss.')
        parser.add_argument('--store_loaded_data', action='store_true', required=False, help='Store the loaded data in main memory during training? This will take way more memory, but way less computing time.')
        parser.add_argument('--augmentation_radius', type=int, default=0, required=False, help='Radius for random scaling at data augmentation. Input: Percentage (e.g. 10)')
        parser.add_argument('--augmentation', type=str, default='None', required=False, help='Choose data augmentation {None}.')
        parser.add_argument('--normalize', type=str, default='global', required=False, help='Choose type of normalization {None, local, global}.')
        parser.add_argument('--n_groups', type=int, default='16', required=False, help='Number of groups for Group Normalization, n_groups = 1 (layer),')
        return parser
