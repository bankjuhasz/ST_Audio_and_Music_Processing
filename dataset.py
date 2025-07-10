import torch
from mir_eval import tempo
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from typing import Dict, List
from utils import show_spectrogram
import random

from constants import *

class DragonDataset(Dataset):
    def __init__(
            self,
            train_dir_all: str = TRAIN_DATA_PATH_ALL,
            train_dir_tempobeats: str = TRAIN_DATA_PATH_TEMPOBEATS,
            train_dir_onsets: str = TRAIN_DATA_PATH_ONSETS,
            sample_rate: int = SAMPLE_RATE,
            hop_length: int = HOP_LENGTH,
            excerpt_length: int = 1000, # in frames
            indices: list[int] | None = None,  # if None, use all, if list, keep only those indices
            augment: bool = False, # flag to enable data augmentation only in the training set
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.excerpt_length = excerpt_length
        self.augment = augment

        self.train_dir_all = train_dir_all
        self.train_dir_tempobeats = train_dir_tempobeats
        self.train_dir_onsets = train_dir_onsets

        # find all file stems by looking for .orig.spect.pt
        stem_paths_all = glob.glob(os.path.join(self.train_dir_all, "*.orig.spect.pt"))
        stem_paths_tempobeats = glob.glob(os.path.join(self.train_dir_tempobeats, "*.orig.spect.pt"))
        stem_paths_onsets = glob.glob(os.path.join(self.train_dir_onsets, "*.orig.spect.pt"))

        # remove .orig.spect.pt suffix
        stem_paths_all = [os.path.basename(p)[:-len(".orig.spect.pt")] for p in stem_paths_all]
        stem_paths_tempobeats = [os.path.basename(p)[:-len(".orig.spect.pt")] for p in stem_paths_tempobeats]
        stem_paths_onsets = [os.path.basename(p)[:-len(".orig.spect.pt")] for p in stem_paths_onsets]

        # mark the directories
        stem_paths_all = [(item, self.train_dir_all) for item in stem_paths_all]
        stem_paths_tempobeats = [(item, self.train_dir_tempobeats) for item in stem_paths_tempobeats]
        stem_paths_onsets = [(item, self.train_dir_onsets) for item in stem_paths_onsets]

        # combine all stems
        # load_annotations + __getitem__ will handle missing annotations
        self.stems = stem_paths_all + stem_paths_tempobeats + stem_paths_onsets

        # this makes all the augmentation logic possible by allowing to differentiate between training, val, and test sets
        if indices is not None:
            self.stems = [self.stems[i] for i in indices]
        self.augment_opts = ["orig", "str08", "str12", "pch+4", "pch-4"]

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        stem, directory = self.stems[idx]

        # determine whether to augment the data
        if self.augment:
            aug_choice = random.choice(self.augment_opts)
        else:
            aug_choice = "orig"
        stem_spect = f"{stem}.{aug_choice}"  # add augmentation suffix to load correct spectrogram

        # load the spectrogram with the augmentation suffix
        spect_path = os.path.join(directory, stem_spect + ".spect.pt")
        feats_original = torch.load(spect_path) # WARNING: 'original' is overloaded here -- it refers to the original format of the labels in this case, not that it has no augmentation
        if aug_choice in ["str08", "str12"]:  # time stretching adds a batch dimension
            feats_original = feats_original.squeeze(0)
        C, F, T = feats_original.shape

        # frame times (in seconds)
        self.frame_length = self.hop_length / self.sample_rate
        self.fps = 1.0 / self.frame_length
        frame_times = np.arange(0, T * self.frame_length, self.frame_length)

        # load onset, beat and tempo annotations
        # these are the original annotations, changes for the chosen augmentation will be applied below
        onsets, beats, tempo_truth_label = self.load_annotations(stem, directory)

        # only stretching affects the labels
        if aug_choice in ["str08", "str12"]:
            # stretch factor is 0.8 or 1.2, so we need to adjust the onsets accordingly
            stretch_factor = 0.8 if aug_choice == "str08" else 1.2

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
            # apply time stretching to original onset annotations
            if aug_choice in ["str08", "str12"]:
                onsets = onsets / stretch_factor # really counterintuitive --> "stretch_factor of 2.0 will double the speed of the audio and halve the length of the audio"
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
            # apply time stretching to original beat annotations
            if aug_choice in ["str08", "str12"]:
                beats = beats / stretch_factor
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

        ### RANDOM EXCERPT SELECTION ###
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
        # apply time stretching to original tempo annotations if applicable
        if aug_choice in ["str08", "str12"]:
            if tempo_truth_label is not None:
                tempo_truth_label[0] = tempo_truth_label[0] * stretch_factor # here we multiply --> "stretch_factor of 2.0 will double the speed of the audio"
                tempo_truth_label[1] = tempo_truth_label[1] * stretch_factor
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
            "audio_name": stem_spect,  # for debugging, to keep track of what augmentation was applied
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
        """ Load annotations for a given observation. """
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