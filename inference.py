import torch
from pathlib import Path
import json

from constants import *
from model import ThreeHeadedDragon
from postprocessing import postp_minimal

def predict_single_example(checkpoint_path: str, spect_path: str, device: str = 'cpu', target: str = 'beat') -> dict:
    model = load_model(checkpoint_path, device)
    with torch.no_grad():
        spect = torch.load(spect_path).unsqueeze(0).to(device)
        pred = model(spect)[target]
    pred = postp_minimal(pred, tolerance=199, inference=True, raw=True)
    return pred

def make_predictions(checkpoint_path, device: str = 'cpu'):
    model = load_model(checkpoint_path, device)
    test_set = load_test_spectrograms()  # returns a list of tensors

    predictions = {}
    for item in test_set:
        stem = item['stem']
        spect = item['spectrogram'].to(device)
        with torch.no_grad():
            spect = spect.unsqueeze(0) # add batch and channel dimensions
            output = model(spect)

        # postprocess each head output
        pred_beats = output['beat'].squeeze(-1)
        pred_onsets = output['onset'].squeeze(-1)


        onsets = postp_minimal(pred_logits=pred_onsets, tolerance=50, inference=True)
        beats = postp_minimal(pred_logits=pred_beats, tolerance=199, inference=True)
        tempo = output['tempo'].cpu().numpy().tolist()

        predictions[stem] = {
            "onsets": onsets,
            "beats": beats,
            "tempo": tempo
        }
    return predictions

def load_test_spectrograms(test_path: str = TEST_DATA_PATH):
    """ Load test spectrograms """
    spectrograms = []
    spect_files = sorted(Path(test_path).glob('*.spect.pt'))
    for spect_path in spect_files:
        spect = torch.load(spect_path)
        stem = spect_path.name.replace('.spect.pt', '')  # get the filename without extension
        spectrograms.append({'stem': stem, 'spectrogram': spect})
    return spectrograms

def load_model(checkpoint_path: str, device: str, freq_dim=FREQ_DIM, stomach_dim=32) -> ThreeHeadedDragon:
    """ Load the trained ThreeHeadedDragon model from a checkpoint. """
    model = ThreeHeadedDragon(
        freq_dim=freq_dim,
        stomach_dim=stomach_dim,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_name = '09_beats_onsets_balanced-loss_plscheduled_beat-focus_e71_frs_tcn.pt'
    ckpt_path = Path(CHECKPOINT_PATH) / ckpt_name
    predictions = make_predictions(ckpt_path)
    # save predictions to a json file
    pred_path = Path(PRED_DATA_PATH) / ckpt_name.replace('.pt', '.pred.json')
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)
