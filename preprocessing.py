import numpy as np
import torch
import torchaudio
from pathlib import Path
from pedalboard import *

from constants import *

def main(data_path=TRAIN_DATA_PATH_ONSETS):
    # main function to process all audio files in the training data directory.

    variants = [ # for data augmentation
        {"name": "orig", "stretch": 1.0, "shift": 0},
        {"name": "str08", "stretch": 0.8, "shift": 0},
        {"name": "str12", "stretch": 1.2, "shift": 0},
        {"name": "pch+4", "stretch": 1.0, "shift": +4},
        {"name": "pch-4", "stretch": 1.0, "shift": -4},
    ]

    for audio_file_path in Path(data_path).glob("*.wav"):
        for v in variants:
            try:
                print(f"Processing {audio_file_path.name} --> {v['name']}")
                process_audio(
                    audio_file_path,
                    data_path,
                    mean=GLOBAL_MEAN, std=GLOBAL_STD,
                    stretch=v["stretch"],
                    shift=v["shift"],
                    suffix=v["name"],
                )
            except Exception as e:
                print(f"Error processing {audio_file_path.name} [{v['name']}]: {e}")


def process_audio(audio_path, in_path, mean=GLOBAL_MEAN, std=GLOBAL_STD, stretch=1.0, shift=0, suffix="orig"):
    """ Load a single audio file, apply augmentations, create a spectrogram, and save it. """

    # loading audio files
    waveform, sample_rate = load_audio(audio_path)

    # data augmentation
    waveform = augment_waveform(waveform, sample_rate, stretch=stretch, shift=shift)

    # create spectrogram
    spectrogram = create_spectrogram(waveform, sample_rate, mean=mean, std=std)

    # save spectrogram
    if spectrogram is not None:
        base = Path(audio_path).stem
        out_name = f"{base}.{suffix}.spect.pt"
        out_path = Path(in_path) / out_name
        save_spectrogram(spectrogram, out_path)


def load_audio(path, dtype="float64", target_sample_rate=SAMPLE_RATE):
    try:
        waveform, samplerate = torchaudio.load(path, channels_first=False)
        waveform = np.asanyarray(waveform.squeeze().numpy(), dtype=dtype)
        if samplerate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=target_sample_rate)
            waveform = resampler(torch.tensor(waveform).float()).numpy()
            print('Resampled audio from {} Hz to {} Hz'.format(samplerate, target_sample_rate))
            samplerate = target_sample_rate
        return waveform, samplerate
    except Exception as e:
        print(f"Error loading audio with torchaudio: {e}")


def augment_waveform(waveform, sample_rate, stretch=1.0, shift=0):
    """ Apply basic data augmentation to the waveform (pitch shift and time stretch). """
    waveform = waveform.astype(np.float32)


    if stretch != 1.0:
        waveform = time_stretch(input_audio=waveform, samplerate=float(sample_rate), stretch_factor=float(stretch))

    if shift != 0:
        pedals = []
        pedals.append(PitchShift(semitones=shift))
        board = Pedalboard(pedals)
        waveform = board(waveform, sample_rate=sample_rate)
        return waveform

    return waveform


def create_spectrogram(
        waveform,
        sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        f_min=30,
        f_max=11000,
        n_mels=128,
        mel_scale="slaney",
        normalized=True,
        power=1,
        log_multiplier=1000,
        mean=GLOBAL_MEAN,
        std=GLOBAL_STD
):
    try:
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
        )(torch.tensor(waveform).float())
        spectrogram = torch.log1p(log_multiplier * spectrogram) # log scaling
        #spectrogram = (spectrogram - mean) / (std + 1e-6) # normalization
        return spectrogram
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None

def save_spectrogram(spectrogram, out_path):
    try:
        torch.save(spectrogram.unsqueeze(0), out_path)
        print(f"Spectrogram saved to {out_path}")
    except Exception as e:
        print(f"Error saving spectrogram: {e}")

def calculate_normalization_stats(datapath):
    """
    Calculate normalization statistics for a list of spectrograms to be taken as constants.

    Args:
        datapath: path to the directory containing spectrogram files.

    Returns:
        tuple: Mean and standard deviation of the spectrograms.
    """
    S_PATH = Path(datapath)
    files = list(S_PATH.glob("*.spect.pt"))

    sum_ = 0.0
    sum_sq = 0.0
    count = 0

    for p in files:
        s = torch.load(p)  # shape [1, T, F] or [T, F]
        s = s.squeeze(0)  # make it [T, F]
        sum_ += s.sum().item()
        sum_sq += (s * s).sum().item()
        count += s.numel()

    global_mean = sum_ / count
    global_var = sum_sq / count - global_mean ** 2
    global_std = global_var ** 0.5

    print(f"GLOBAL_MEAN = {global_mean:.6f}")
    print(f"GLOBAL_STD  = {global_std:.6f}")

if __name__ == "__main__":
    #calculate_normalization_stats(TRAIN_DATA_PATH)
    main()
