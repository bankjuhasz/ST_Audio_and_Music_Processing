import torch
from mir_eval import tempo
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from typing import Dict, List
from utils import show_spectrogram

from constants import *

class DragonDataset(Dataset):
    def __init__(
            self,
            train_dir_all: str = TRAIN_DATA_PATH_ALL,
            train_dir_tempobeats: str = TRAIN_DATA_PATH_TEMPOBEATS,
            train_dir_onsets: str = TRAIN_DATA_PATH_ONSETS,
            sample_rate: int = SAMPLE_RATE,
            hop_length: int = HOP_LENGTH,
            excerpt_length: int = 1000 # in frames
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.excerpt_length = excerpt_length

        self.train_dir_all = train_dir_all
        self.train_dir_tempobeats = train_dir_tempobeats
        self.train_dir_onsets = train_dir_onsets

        # find all file stems by looking for .spect.pt
        stem_paths_all = glob.glob(os.path.join(self.train_dir_all, "*.spect.pt"))
        stem_paths_tempobeats = glob.glob(os.path.join(self.train_dir_tempobeats, "*.spect.pt"))
        stem_paths_onsets = glob.glob(os.path.join(self.train_dir_onsets, "*.spect.pt"))

        # remove .spect.pt suffix
        stem_paths_all = [os.path.basename(p)[:-len(".spect.pt")] for p in stem_paths_all]
        stem_paths_tempobeats = [os.path.basename(p)[:-len(".spect.pt")] for p in stem_paths_tempobeats]
        stem_paths_onsets = [os.path.basename(p)[:-len(".spect.pt")] for p in stem_paths_onsets]

        # mark the directories
        stem_paths_all = [(item, self.train_dir_all) for item in stem_paths_all]
        stem_paths_tempobeats = [(item, self.train_dir_tempobeats) for item in stem_paths_tempobeats]
        stem_paths_onsets = [(item, self.train_dir_onsets) for item in stem_paths_onsets]

        # combine all stems
        # load_annotations will handle missing annotations
        self.stems = stem_paths_all + stem_paths_tempobeats + stem_paths_onsets

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        stem, directory = self.stems[idx]

        # load features
        spect_path = os.path.join(directory, stem + ".spect.pt")
        feats_original = torch.load(spect_path) # expect [C, F, T]
        C, F, T = feats_original.shape

        # frame times (in seconds)
        self.frame_length = self.hop_length / self.sample_rate
        self.fps = 1.0 / self.frame_length
        frame_times = np.arange(0, T * self.frame_length, self.frame_length)

        # load onset, beat and tempo annotations
        onsets, beats, tempo_truth_label = self.load_annotations(stem, directory)

        ### BEATS AND ONSETS ###

        # build frame-wise labels
        onset_label = np.zeros(shape=(T,), dtype=np.float32)
        beat_label = np.zeros(shape=(T,), dtype=np.float32)
        # build frame-wise loss weights
        onset_weight = np.ones(shape=(T,), dtype=np.float32)
        beat_weight = np.ones(shape=(T,), dtype=np.float32)

        if onsets is not None: # if onsets are present, otherwise the label will be all zeros
            # build frame-wise masks for missing annotations
            onset_mask = np.ones(shape=(T,), dtype=np.float32)
            for t in onsets:
                i = np.searchsorted(frame_times, t)
                i -= 1
                if i < T:
                    onset_label[i] = 1.0
                    onset_weight[i] = 50
                    # mark neighbors as positive, half weight
                    '''
                    if i - 1 >= 0:
                        onset_label[i - 1] = 1.0
                        onset_weight[i - 1] = 25
                    if i + 1 < T:
                        onset_label[i + 1] = 1.0
                        onset_weight[i + 1] = 25'''
        else:
            onset_mask = np.zeros(shape=(T,), dtype=np.float32)

        if beats is not None: # if beats are present, otherwise the label will be all zeros
            # build frame-wise masks for missing annotations
            beat_mask = np.ones(shape=(T,), dtype=np.float32)
            for t in beats:
                i = np.searchsorted(frame_times, t)
                i -= 1
                if i < T:
                    beat_label[i] = 1.0
                    beat_weight[i] = 4
                    # mark neighbors as positive, half weight
                    '''if i - 1 >= 0:
                        beat_label[i - 1] = 1.0
                        beat_weight[i - 1] = 2
                    if i + 1 < T:
                        beat_label[i + 1] = 1.0
                        beat_weight[i + 1] = 2'''
        else:
            beat_mask = np.zeros(shape=(T,), dtype=np.float32)

        # --- Random excerpt selection ---
        if T > self.excerpt_length:
            start = np.random.randint(0, T - self.excerpt_length + 1)
            end = start + self.excerpt_length
            feats = feats_original[:, :, start:end]
            onset_label = onset_label[start:end]
            beat_label = beat_label[start:end]
            onset_weight = onset_weight[start:end]
            beat_weight = beat_weight[start:end]
            onset_mask = onset_mask[start:end]
            beat_mask = beat_mask[start:end]
            if beats is not None:
                beat_truth_times = beats - start * self.frame_length # shift beat times to the excerpt start
                beat_truth_times = beat_truth_times[beat_truth_times >= 0]  # remove negative times --> removed beats before the excerpt start
                beat_truth_times = beat_truth_times[beat_truth_times < self.excerpt_length * self.frame_length]  # remove beats after the excerpt end
            else:
                beat_truth_times = np.array([])
            if onsets is not None:
                onset_truth_times = (onsets - start * self.frame_length)
                onset_truth_times = onset_truth_times[onset_truth_times >= 0]
                onset_truth_times = onset_truth_times[onset_truth_times < self.excerpt_length * self.frame_length]
            else:
                onset_truth_times = np.array([])
        else:
            # Pad if too short
            pad_width = self.excerpt_length - T
            feats = torch.nn.functional.pad(feats_original, (0, pad_width))
            onset_label = np.pad(onset_label, (0, pad_width), mode='constant', constant_values=0.0)
            beat_label = np.pad(beat_label, (0, pad_width), mode='constant', constant_values=0.0)
            onset_weight = np.pad(onset_weight, (0, pad_width), mode='constant', constant_values=0.0)
            beat_weight = np.pad(beat_weight, (0, pad_width), mode='constant', constant_values=0.0)
            beat_mask = np.pad(beat_mask, (0, pad_width), mode='constant', constant_values=0.0)
            onset_mask = np.pad(onset_mask, (0, pad_width), mode='constant', constant_values=0.0)
            beat_truth_times = beats.copy() if beats is not None else np.array([])
            onset_truth_times = onsets.copy() if onsets is not None else np.array([])


        ### TEMPO ###
        tempo_label, tempo_mask = self.make_tempo_label(tempo_truth_label, max_bpm=300)  # [300,]
        if tempo_truth_label is None:
            tempo_truth_label = np.array([])

        return {
            "features": feats,  # [C, F, T]
            "onset_label": torch.from_numpy(onset_label),  # [T]
            "beat_label": torch.from_numpy(beat_label),  # [T]
            "tempo_label": torch.from_numpy(tempo_label), # [300]
            "onset_truth_times": torch.from_numpy(onset_truth_times),
            "beat_truth_times": torch.from_numpy(beat_truth_times),
            "tempo_truth_label": torch.from_numpy(tempo_truth_label),  # [3] or None
            "onset_weight": torch.from_numpy(onset_weight),  # [T]
            "beat_weight": torch.from_numpy(beat_weight),  # [T]
            "beat_mask": torch.from_numpy(beat_mask),  # [T]
            "onset_mask": torch.from_numpy(onset_mask),  # [T]
            "tempo_mask": torch.from_numpy(tempo_mask),  # [T]
            "audio_name": stem,  # for debugging
        }

    @staticmethod
    def make_tempo_label(tempo: np.ndarray, max_bpm: int = 300) -> np.ndarray:
        """Convert (low, high, proportion) label to 300 dimensional probability vector."""
        if tempo is None or len(tempo) != 3:
            # no tempo label --> return zero mask and zero label vector
            tempo_mask = np.zeros(max_bpm, dtype=np.float32)
            tempo_label = np.zeros(max_bpm, dtype=np.float32)
            return tempo_label, tempo_mask

        low, high, prop = tempo[0], tempo[1], tempo[2]

        low_bin = int(np.round(low))
        high_bin = int(np.round(high))
        low_bin = max(1, min(max_bpm, low_bin)) # just for safety
        high_bin = max(1, min(max_bpm, high_bin)) # just for safety

        low_idx = low_bin - 1  # zero-based index
        high_idx = high_bin - 1

        # building the label vector
        tempo_label = np.zeros(max_bpm, dtype=np.float32)
        if low_idx == high_idx:
            # only one tempo, all annotators agree
            tempo_label[low_idx] = 1.0
        else:
            # two tempi and a proportion
            tempo_label[low_idx] = prop
            tempo_label[high_idx] = 1-prop

        # building the mask for the loss fn --> tempo label is present --> all ones
        tempo_mask = np.ones(max_bpm, dtype=np.float32)

        return tempo_label, tempo_mask # both [300,]


    def load_annotations(self, stem: str, directory: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load annotations for a given observation.
        Based on the directory, it may return None if annotations are missing.
        Args:
            stem (str): stem of the target file
            directory (str): directory where the annotations are stored
        Returns:
            tuple: (onsets, beats, tempo)
        """

        # onsets
        if directory == self.train_dir_all or directory == self.train_dir_onsets: # onset annotations present
            ext = ".onsets.gt"
            path = os.path.join(directory, stem + ext)
            try:
                arr = np.loadtxt(path, dtype=np.float32)
                onsets = np.atleast_1d(arr)
            except Exception as e:
                print(f"Could not load onsets from {path}: {e}")
                onsets = None
        else: # no onset label for this example
            onsets = None

        # beats and tempo
        if directory == self.train_dir_all or directory == self.train_dir_tempobeats: # beat annotations present
            # beats
            ext = ".beats.gt"
            path = os.path.join(directory, stem + ext)
            try:
                arr = np.loadtxt(path, dtype=np.float32)
                beats = np.atleast_1d(arr)[:, 0] # taking first col --> we ignore beat counts
            except Exception as e:
                print(f"Could not load beats from {path}: {e}")
                beats = None

            # tempo
            ext = ".tempo.gt"
            path = os.path.join(directory, stem + ext)
            try:
                arr = np.loadtxt(path, dtype=np.float32)
                arr = np.atleast_1d(arr)
                if arr.size == 1:
                    # only one tempo, all annotators agree
                    # altering the annotation format to [tempo, tempo, 1.0]
                    tempo = np.array([arr[0], arr[0], 1], dtype=np.float32)
                elif arr.size == 3:
                    '''
                    ### this version only uses the more confident tempo label ###

                    # two tempi and a proportion
                    slow, fast, prop = arr
                    if prop >= 0.5:
                        tempo = float(slow)
                    else:
                        tempo = float(fast)
                    '''
                    tempo = arr
                else:
                    # unexpected format
                    print(f"Warning: Unexpected tempo format in {path}: {arr}")
                    tempo = None
            except Exception as e:
                print(f"Could not load tempo from {path}: {e}")
                tempo = None

        else: # no beat and tempo label for this example
            beats = None
            tempo = None

        return onsets, beats, tempo


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # pad all T to the max length in this batch
    Ts = [s["features"].shape[-1] for s in batch]
    T_max = max(Ts)
    B = len(batch)
    C, F, _ = batch[0]["features"].shape # infer C and F from the first item


    feats = torch.zeros((B, C, F, T_max), dtype=torch.float32)
    onsets = torch.zeros((B, T_max), dtype=torch.float32)
    beats = torch.zeros((B, T_max), dtype=torch.float32)
    tempi = torch.zeros((B, 300), dtype=torch.float32)
    masks = torch.zeros((B, T_max), dtype=torch.bool)  # for padding
    onset_weight = torch.zeros((B, T_max), dtype=torch.float32)
    beat_weight = torch.zeros((B, T_max), dtype=torch.float32)
    beat_mask = torch.zeros((B, T_max), dtype=torch.float32)
    onset_mask = torch.zeros((B, T_max), dtype=torch.float32)
    tempo_mask = torch.zeros((B, 300), dtype=torch.float32)

    onsets_truth_times = []
    beats_truth_times = []
    tempo_truth_label = []
    audio_names = []
    for i, s in enumerate(batch):
        t = s["features"].shape[-1]
        feats[i, :, :, :t] = s["features"]
        onsets[i, :t] = s["onset_label"]
        beats[i, :t] = s["beat_label"]
        tempi[i, :300] = s["tempo_label"] # assuming max_bpm is 300
        masks[i, :t] = True
        onset_weight[i, :t] = s["onset_weight"]
        beat_weight[i, :t] = s["beat_weight"]
        onsets_truth_times.append(s["onset_truth_times"])
        beats_truth_times.append(s["beat_truth_times"])
        tempo_truth_label.append(s["tempo_truth_label"])
        beat_mask[i, :t] = s["beat_mask"]
        onset_mask[i, :t] = s["onset_mask"]
        tempo_mask[i, :300] = s["tempo_mask"] # assuming max_bpm is 300
        audio_names.append(s["audio_name"])

    ### DEBUGGING ###
    '''
    for i, s in enumerate(batch):
        feats_ = feats[i]
        spect_ = feats_.squeeze(0)
        beat_label_ = beats[i]
        show_spectrogram(spect_)
        show_spectrogram(spect_, beat_label_)
    '''

    return {
        "features": feats, # [B, C, F, T_max]
        "beat_label": beats,  # [B, T_max]
        "onset_label": onsets, # [B, T_max]
        "tempo_label": tempi, # [B, 300]
        "beat_truth_times": beats_truth_times,
        "onset_truth_times": onsets_truth_times,
        "tempo_truth_label": tempo_truth_label,
        "beat_weight": beat_weight,  # [B, T_max]
        "onset_weight": onset_weight, # [B, T_max]
        "beat_mask": beat_mask,  # [B, T_max]
        "onset_mask": onset_mask,  # [B, T_max]
        "tempo_mask": tempo_mask,  # [B, 300]
        "padding_mask": masks,  # [B, T_max]
        "audio_name": audio_names,  # [B]
    }

    return batch