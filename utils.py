import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

def show_spectrogram(spectrogram, beat_labels=None, labels_as_time=False):
    # convert to numpy if it's a torch tensor
    if hasattr(spectrogram, "numpy"):
        spectrogram = spectrogram.numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.title('Mel Spectrogram')

    if beat_labels is not None and not labels_as_time:
        # convert to numpy if tensor
        if hasattr(beat_labels, "numpy"):
            beat_labels = beat_labels.numpy()
        # Find indices where beat label is positive
        beat_indices = np.where(beat_labels > 0)[0]
        for t in beat_indices:
            plt.axvline(x=t, color='red', linestyle='--', linewidth=1, alpha=0.7)
    elif beat_labels is not None and labels_as_time:
        beat_labels = beat_labels * 100
        for t in beat_labels:
            plt.axvline(x=t, color='red', linestyle='--', linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.show()

