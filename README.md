# AnMP Challenge - Onset, Beat, and Tempo Detection

**Bank JUHASZ | K11918898 | Team J | SS2025**

---

## Short Description

The Three-Headed Dragon for beats, onsets and tempo. See the report for a detailed description. My original plan was to
include a neat CLI, but unfortunately, I didn't get to it.

## Usage

The training script assumes precomputed spectrograms (and their augmented forms) under `data`. Training can be started
by running `training.py`. All hyperparameters must be changed in there, or in `loss.py`, `model.py`, `postprocessing.py`.
Given more time, and were this a serious, published work, this would obviously be much more tidy.

As for making predictions, `inference.py` is relatively straightforward. There is also no CLI here, but all it needs is
`ckpt_name` under `if __name__ == '__main__':` to select the desired model in the `checkpoints` folder. This script
creates a joint prediction json file in the required format, and saves it under `predictions`.

## Potential Issues

A small caveat is the `madmom` package, which was devilishly difficult to make work. I suspect that if you manage to
install _any_ working version of it, the script will run successfully, but I had to install a very specific, non-official
version which fixed a specific dependency issue. In case of issues relating to this, please try to use [this](https://github.com/CPJKU/madmom/pull/548/files).
If memory serves right, I installed this using `pip install git+https://github.com/CPJKU/madmom.git@refs/pull/548/head`.