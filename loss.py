from constants import *
import torch
import torch.nn.functional as F
from postprocessing import postp_minimal
import numpy as np
import mir_eval


# loss computation
def compute_loss(model, outputs, batch, epoch=0, stage='train'):
    """Compute the loss for the model outputs and calculate metrics every n epochs."""

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics_every_n_epoch = 1

    metrics = {}

    # unpack batch
    y_on = batch['onset_label'].to(DEVICE)
    y_be = batch['beat_label'].to(DEVICE)
    y_tm = batch['tempo_label'].to(DEVICE)
    padding_mask = batch['padding_mask'].to(DEVICE)
    weight_on = batch['onset_weight'].to(DEVICE)
    weight_be = batch['beat_weight'].to(DEVICE)
    beat_truth_times = batch['beat_truth_times']
    onset_truth_times = batch['onset_truth_times']
    beat_mask = batch['beat_mask'].to(DEVICE)
    onset_mask = batch['onset_mask'].to(DEVICE)
    tempo_mask = batch['tempo_mask'].to(DEVICE)

    # getting predictions
    pred_on = outputs['onset'].squeeze(-1)
    pred_be = outputs['beat'].squeeze(-1)
    pred_tm = outputs['tempo'].squeeze(-1)

    '''
    # previous version --> non-fixed sized examples
    #loss_on = F.binary_cross_entropy(pred_on, y_on, weight=weight_on, reduction='none')
    #loss_be = F.binary_cross_entropy(pred_be, y_be, weight=weight_be, reduction='none')
    # zero out padded frames
    #loss_on = loss_on * padding_mask  # broadcasting [B,T] * [B,T]
    #loss_be = loss_be * padding_mask
    # avg over real frames
    #loss_on = loss_on.sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1)
    #loss_be = loss_be.sum(dim=1) / padding_mask.sum(dim=1).clamp(min=1)
    '''

    ### BEATS AND ONSETS ###

    # shift-tolerant BCE loss from beat_this
    criterion = ShiftTolerantBCELoss(pos_weight=50)
    loss_be = criterion(pred_be, y_be, padding_mask)
    loss_on = criterion(pred_on, y_on, padding_mask)
    # the (non-reduced) tensor returned by the STBCELoss has shorter sequence lengths than T, so the original masks
    # are not directly compatible. But the masks can be shortened, as they have a fixed value for each a specific row.
    beat_mask = beat_mask[:, :loss_be.shape[1]]
    onset_mask = onset_mask[:, :loss_on.shape[1]]
    # apply the beat and onset masks --> zero out losses for examples without annotations
    loss_be = loss_be * beat_mask
    loss_on = loss_on * onset_mask

    # calculating f1 scores for beats and onsets
    if epoch % metrics_every_n_epoch == 0:
        # beats
        postp_be = postp_minimal(pred_be, padding_mask, tolerance=199) # postprocess to get beat predictions in seconds
        beat_f1 = []
        for ref_beats, est_beats in zip(beat_truth_times, postp_be):
            if len(ref_beats) == 0:
                continue
            f1 = mir_eval.onset.f_measure(ref_beats, est_beats, window=0.05)
            beat_f1.append(f1)
        beat_f1 = np.mean(beat_f1) if beat_f1 else 0.0

        # onsets
        postp_on = postp_minimal(pred_on, padding_mask, tolerance=50)  # postprocess to get onset predictions in seconds
        onset_f1 = []
        for ref_onsets, est_onsets in zip(onset_truth_times, postp_on):
            if len(ref_onsets) == 0:
                continue
            f1 = mir_eval.beat.f_measure(ref_onsets, est_onsets, f_measure_threshold=0.05)
            onset_f1.append(f1)
        onset_f1 = np.mean(onset_f1) if onset_f1 else 0.0

        # tempo TODO
        '''pred_tm = outs['tempo'].detach().cpu().numpy()
        gt_tm = batch['tempo_label'].cpu().numpy()
        for pred, gt in zip(pred_tm, gt_tm):
            if gt.any() > 0:
                tempo_accs.append(mir_eval.tempo.detection(gt[:2], gt[2], np.clip(pred[:2], 0, None), tol=0.08))'''

        metrics = {
            'beat_f1': beat_f1,
            'onset_f1': onset_f1,
            'tempo_mae': 0.0,  # TODO
        }

    # tempo loss
    #max_bpm = 200.0
    #pred_tm = outputs['tempo']  # [B, 3]
    #y_tm_norm = y_tm[:, :2] / max_bpm  # [B, 2]
    #pred_tm_norm = pred_tm[:, :2] / max_bpm  # [B, 2]

    # weight (we do not normalize)
    #y_weight = y_tm[:, 2]  # [B]
    #pred_weight = pred_tm[:, 2]  # [B]

    # mse loss for tempi
    #loss_tm = F.mse_loss(pred_tm_norm, y_tm_norm, reduction='none').mean(dim=1)  # [B]
    # mse loss for weight (bce also possible)
    #loss_weight = F.mse_loss(pred_weight, y_weight, reduction='none')  # [B]

    # combine tempo and weight losses
    #loss_tempo_total = loss_tm + loss_weight

    '''
    # masked sum
    #total = m_o * loss_on + m_b * loss_be + m_t * loss_tm
    # normalize per-example
    #norm = (m_o + m_b + m_t).clamp(min=1e-8)
    '''
    # turn raw_log_var → positive log_var
    log_var_on = F.softplus(model.raw_log_var_on)
    log_var_be = F.softplus(model.raw_log_var_be)
    #log_var_tm = F.softplus(model.raw_log_var_tm)

    # convert log‐vars --> precision
    precision_on = torch.exp(-log_var_on)
    precision_be = torch.exp(-log_var_be)
    #precision_tm = torch.exp(-log_var_tm)

    weighted_summed_loss = (#loss_be + loss_on
        precision_on * loss_on + log_var_on +
        precision_be * loss_be + log_var_be
        #precision_tm * loss_tm + log_var_tm
    )
    #weighted_summed_loss = 3*loss_on + loss_be

    return weighted_summed_loss.mean(), metrics



class ShiftTolerantBCELoss(torch.nn.Module):
    """
    BCE loss variant for sequence labeling that tolerates small shifts between
    predictions and targets. This is accomplished by max-pooling the
    predictions with a given tolerance and a stride of 1, so the gradient for a
    positive label affects the largest prediction in a window around it.
    Expects predictions to be given as logits, and accepts an optional mask
    with zeros indicating the entries to ignore. Note that the edges of the
    sequence will not receive a gradient, as it is assumed to be unknown
    whether there is a nearby positive annotation.

    Args:
        pos_weight (float): Weight for positive examples compared to negative
            examples (default: 1)
        tolerance (int): Tolerated shift in time steps in each direction
            (default: 3)
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 3):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, x: torch.Tensor, factor: int = 1):
        if self.tolerance == 0:
            return x
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x: torch.Tensor, factor: int = 1):
        return x[..., factor * self.tolerance : -factor * self.tolerance or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # spread preds and crop targets to match
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)
        # ignore around the positive targets
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:  # consider padding and no-downbeat mask
            look_at = look_at * self.crop(mask, factor=2)
        # compute loss
        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
            reduction='none',
        )

