from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.onsets import OnsetPeakPickingProcessor

def madmom_onsets(onset_activations, threshold=0.5, combine=0.03, pre_max=3, post_max=3):
    """ apply madmom's OnsetPeakPickingProcessor to onset activations """
    peak_picker = OnsetPeakPickingProcessor(
        threshold=threshold,
        combine=combine,
        pre_max=0,
        post_max=0
    )
    onset_times = []
    for onset in onset_activations:
        #onset_list = list(onset[0])
        onset_times.append(peak_picker(onset))
    return onset_times

def madmom_dbn_beats(beat_activations, fps=100, min_bpm=30, max_bpm=300):
    """ apply madmom's DBNBeatTrackingProcessor to beat activations """
    processor = DBNBeatTrackingProcessor(fps=fps, min_bpm=min_bpm, max_bpm=max_bpm, transition_lambda=500, online=False)
    beat_times = processor(beat_activations)
    return beat_times


def postp_tempo(pred_logits: torch.Tensor, neighborhood: float=0.2):
    """ turning tempo logits into probabilities, then extracting two tempo predictions --> for a single piece """
    probs_tm = F.softmax(pred_logits, dim=0)
    top1_idx = torch.argmax(probs_tm).item()
    b1 = top1_idx + 1  # from idx to bpm

    # a bit of trickery: I'm allowed to submit two estimates, but +-8% is still considered correct. therefore, I calculate
    # a neighborhood around my best guess and take the second best guess OUTSIDE of that neighborhood, as two very close
    # predictions cancel each other out.
    low_cut = b1 * (1.0 - neighborhood)
    min_bin = int(torch.floor(torch.tensor(low_cut)).item())
    high_cut = b1 * (1.0 + neighborhood)
    max_bin = int(torch.ceil(torch.tensor(high_cut)).item())
    min_bin = max(1, min(300, min_bin)) # clamp to [1,300]
    max_bin = max(1, min(300, max_bin))
    min_idx = min_bin - 1  # from bpm to idx
    max_idx = max_bin - 1
    mask = torch.ones_like(probs_tm, dtype=torch.bool)  # [300,]
    mask[min_idx: max_idx + 1] = False

    masked_scores = probs_tm.clone()
    masked_scores[~mask] = -1e9 # for values inside the neighborhood, set to very low value
    second_idx = torch.argmax(masked_scores).item()
    b2 = second_idx + 1

    # sort to conform to the submission requirements
    b2, b1 = min(b1, b2), max(b1, b2)

    return np.array([b2, b1]) # mir_eval expects np.ndarray


def postp_minimal(pred_logits, padding_mask=None, tolerance=70, inference=False, raw=False):
    """ Taken from Beat This! """
    tolerance = int(tolerance / 10)
    if padding_mask is None:
        padding_mask = torch.ones_like(pred_logits, dtype=torch.bool)
    # set padded elements to -1000 (= probability zero even in float64) so they don't influence the maxpool
    pred_logits = pred_logits.masked_fill(~padding_mask, -1000)

    # pick maxima within +/- tolerance ms
    pred_peaks = pred_logits.masked_fill(
        pred_logits != F.max_pool1d(pred_logits, tolerance, 1, int((tolerance-1)/2)), -1000
    )
    # keep maxima with over 0.5 probability (logit > 0)
    pred_peaks = pred_peaks > 0
    if raw:
        return pred_peaks

    # run the piecewise operations
    postp_peaks = [
        _postp_minimal_item(pp, pm)
        for pp, pm in zip(pred_peaks, padding_mask)
    ]
    if inference:
        # flatten to a single list of times (floats) --> inference one item at a time
        return np.concatenate(postp_peaks).tolist()

    return postp_peaks

def _postp_minimal_item(pred_peaks, mask):
    """
    Taken from Beat This!
    Function to compute the operations that must be computed piece by piece, and cannot be done in batch.
    """
    # unpad the predictions by truncating the padding positions
    pred_peaks = pred_peaks[mask]
    # pass from a boolean array to a list of times in frames.
    pred_frame = torch.nonzero(pred_peaks).cpu().numpy()[:, 0]
    # remove adjacent peaks
    pred_frame = deduplicate_peaks(pred_frame, width=1)

    # convert from frame to seconds
    pred_time = pred_frame / 100 # assuming fps is 100

    # remove duplicate downbeat times (if some db were moved to the same position)
    pred_time = np.unique(pred_time)

    return pred_time

def deduplicate_peaks(peaks, width=1) -> np.ndarray:
    """
    Taken from Beat This!
    Replaces groups of adjacent peak frame indices that are each not more
    than `width` frames apart by the average of the frame indices.
    """
    result = []
    peaks = map(int, peaks)  # ensure we get ordinary Python int objects
    try:
        p = next(peaks)
    except StopIteration:
        return np.array(result)
    c = 1
    for p2 in peaks:
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c  # update mean
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)