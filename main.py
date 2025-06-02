import argparse
import torch
from pathlib import Path
from constants import *
from utils import show_spectrogram
import numpy as np

from inference import *




if __name__ == '__main__':
    spect_files = sorted(Path(TRAIN_DATA_PATH_ALL).glob('train20.spect.pt'))
    for i, spect_path in enumerate(spect_files, 1):
        spect = torch.load(spect_path)
        spect = spect.squeeze(0)
        print(f"[{i}] {spect_path.name} — Spectrogram shape: {spect.shape}")


        pred = predict_single_example('checkpoints/test_beats_only_extended_train_e47_frs_tcn.pt', spect_path)

        show_spectrogram(spect)

