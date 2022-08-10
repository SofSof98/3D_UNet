from .base_options import BaseOptions

class PredictOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--pet_file', type=str, required=True, help='Which PET file shall be used?')
        parser.add_argument('--prostate_contour', type=str, required=True, help='Which prostate contour?')
        parser.add_argument('--trained_model', type=str, required=True, help='Which trained network?')
        parser.add_argument('--prediction_name', type=str, required=False, default="results/" + "_lesions_prediction.nrrd", help='Name of the prediction.')
        parser.add_argument('--normalize', type=str, default='global', required=False, help='Choose type of normalization {None, local, global}.')
        
        return parser