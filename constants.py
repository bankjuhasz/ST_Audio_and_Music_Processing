TRAIN_DATA_PATH = "data/train/"
TRAIN_DATA_PATH_ALL = "data/train_all_annotations/"
TRAIN_DATA_PATH_TEMPOBEATS = "data/train_extra_tempobeats/"
TRAIN_DATA_PATH_ONSETS = "data/train_extra_onsets/"
TEST_DATA_PATH = "data/test/"
PRED_DATA_PATH = "predictions/"
CHECKPOINT_PATH = "checkpoints/"

# Constants for spectrogram normalization -- calculated once from the training set
GLOBAL_MEAN = 3.589251
GLOBAL_STD  = 1.684540

# Audio processing
HOP_LENGTH  = 441
SAMPLE_RATE = 44100
N_FFT       = 1024
FREQ_DIM    = 128