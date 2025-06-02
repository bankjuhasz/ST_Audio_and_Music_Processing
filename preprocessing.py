import numpy as np
import torch
import torchaudio
from pathlib import Path
from constants import *

def main():
    # main function to process all audio files in the training data directory.

    for audio_file_path in Path(TEST_DATA_PATH).glob("*.wav"):
        try:
            print(f"Processing {audio_file_path.name}")
            process_audio(audio_file_path, out_path=Path(PRED_DATA_PATH) / audio_file_path.name, mean=GLOBAL_MEAN, std=GLOBAL_STD)
        except Exception as e:
            print(f"Error processing {audio_file_path.name}: {e}")


def process_audio(audio_path, out_path, mean=GLOBAL_MEAN, std=GLOBAL_STD):
    """
    Load a single audio file, apply augmentations, create a spectrogram, and save it.

    Args:
        audio_path (str or Path): Path to the audio file.
        out_path (str or Path): Path to save the spectrogram.
    """
    try:
        # loading audio files
        waveform, sample_rate = load_audio(audio_path)
        # data augmentation
        # TODO: Implement data augmentation

        # create spectrogram
        spectrogram = create_spectrogram(waveform, sample_rate, mean=mean, std=std)

        # save spectrogram
        if spectrogram is not None:
            out_path = Path(audio_path).with_suffix('.spect.pt')
            save_spectrogram(spectrogram, out_path)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


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

def augment_waveform(waveform, sample_rate, shift=0, stretch=1.0):
    """
    Apply data augmentation to the waveform.

    Args:
        waveform (np.ndarray): The audio waveform.
        sample_rate (int): The sample rate of the audio.
        shift (int): Number of samples to shift the waveform.
        stretch (float): Factor by which to stretch the waveform.

    Returns:
        np.ndarray: Augmented waveform.
    """
    #TODO: Implement data augmentation logic

    return None

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
