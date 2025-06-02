import torch
from dataset import DragonDataset, collate_fn
from torch.utils.data import random_split, DataLoader, Subset
import wandb
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constants import *
from model import ThreeHeadedDragon
from loss import compute_loss

### Hyperparameters ###
BATCH_SIZE    = 64
NUM_WORKERS   = 0
FREQ_DIM      = 128
STOMACH_DIM   = 32
CONV_OUT_CH   = 32
LR            = 1e-3
NUM_EPOCHS    = 1000
PATIENCE      = 20
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
SEND_TO_WANDB = True
CHECKPOINT_PATH = Path(CHECKPOINT_PATH)
CHECKPOINT_NAME_STEM = '09_beats_onsets_balanced-loss_plscheduled_beat-focus_e{epoch}_frs_tcn.pt'
SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if SEND_TO_WANDB:
    wandb.init(
        project="ST_Audio_and_Music_Processing",
        name="Dragon_" + CHECKPOINT_NAME_STEM.format(epoch=''),
        config={
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "hop_length": HOP_LENGTH,
            "sample_rate": SAMPLE_RATE,
            "freq_dim": FREQ_DIM,
            "stomach_dim": STOMACH_DIM,
            "lr": LR,
            "num_epochs": NUM_EPOCHS,
        }
    )

### Prepare dataset and loaders ###
dataset = DragonDataset(
    sample_rate=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
)
# data split
train_size = int(0.7 * len(dataset))
val_size   = (len(dataset) - train_size) // 2
test_size = (len(dataset) - train_size) - val_size
print(f"Dataset size: {len(dataset)}")
print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

# DEBUG - Select the first batch (e.g., 16 samples)
#fixed_indices = list(range(4))
#fixed_subset = Subset(train_ds, fixed_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

# class weights for BCE loss
#all_on = np.concatenate([s['onset_label'].numpy() for s in train_ds])
#n_pos, n_neg = all_on.sum(), all_on.size - all_on.sum()
#pos_weight_on = torch.tensor([(n_neg / ((n_pos/3)+1e-8))/1], device=DEVICE)

#all_be = np.concatenate([s['beat_label'].numpy() for s in train_ds])
#n_pos, n_neg = all_be.sum(), all_be.size - all_be.sum()
#pos_weight_be = torch.tensor([(n_neg / ((n_pos/3)+1e-8))/1], device=DEVICE)

#print('Positive weights for BCE loss:')
#print(f'Onset: {pos_weight_on.item():.4f}, Beat: {pos_weight_be.item():.4f}')

### Initialize model, optimizer ###
print(f'Device: {DEVICE}')
print(f'Initializing model...')
model = ThreeHeadedDragon(
    freq_dim=FREQ_DIM,
    stomach_dim=STOMACH_DIM,
).to(DEVICE)
print(model)
if SEND_TO_WANDB:
    wandb.watch(model, log="gradients", log_freq=20) # for debugging purposes
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}")

#optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)


#####################
### Training loop ###
#####################

best_val_loss = float('inf')
best_beat_f1 = 0.0
epochs_no_improve = 0
best_epoch = None
best_model_state = None

# if you've ever wondered how to train a dragon, here it is:
for epoch in range(1, NUM_EPOCHS+1):

    ### Training ###

    model.train()
    train_loss = 0.0
    train_beat_f1s = 0.0
    train_onset_f1s = 0.0
    train_tempo_maes = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        feats = batch['features'].to(DEVICE)

        ### DEBUG ###
        #feat = model.rearrange_for_transformer(model.body(model.stomach(feats)))
        #print("feat min/max:", feat.min().item(), feat.max().item())

        outs = model(feats)

        ### DEBUG ###
        #for layer_name, activation in model._acts.items():
        #    print(f"{layer_name:20s} --> mean={activation.mean():.4f}, std={activation.std():.4f}, shape={activation.shape}")

        loss, metrics = compute_loss(model, outs, batch, epoch, stage='train')
        optimizer.zero_grad()

        ### DEBUG ###
        #preds_tm = opts['tempo']
        #print("pred_tm min/max:", preds_tm.min().item(), preds_tm.max().item())

        loss.backward()

        # log all gradient norms to W&B
        #if SEND_TO_WANDB:
        #    for name, p in model.named_parameters():
        #        if p.grad is not None:
        #            wandb.log({f"grad_norm/{name}": p.grad.norm().item()})

        ### DEBUG ###
        # visualize after N epochs
        if epoch == 100 or epoch == 500 or epoch == 1000:
            with torch.no_grad():
                x = batch["features"].to(DEVICE)
                feats_ = model.frontendstomach(x)
                from utils import show_spectrogram
                for i in range(3):
                    spect = feats_.cpu()
                    show_spectrogram(spect[i])
            break

        train_beat_f1, train_onset_f1, train_tempo_mae = metrics['beat_f1'], metrics['onset_f1'], metrics['tempo_mae']

        optimizer.step()
        train_loss += loss.item() * feats.size(0)
        train_beat_f1s += train_beat_f1 * feats.size(0)
        train_onset_f1s += train_onset_f1 * feats.size(0)
        train_tempo_maes += train_tempo_mae * feats.size(0)

    train_loss /= len(train_ds)
    train_beat_f1s /= len(train_ds)
    train_onset_f1s /= len(train_ds)
    train_tempo_maes /= len(train_ds)

    ### Validation ###
    model.eval()
    val_loss = 0.0
    val_beat_f1s = 0.0
    val_onset_f1s = 0.0
    val_tempo_maes = 0.0
    with torch.no_grad():
        for batch in val_loader:
            feats = batch['features'].to(DEVICE)
            outs = model(feats)
            loss, metrics = compute_loss(model, outs, batch, epoch, stage='val')
            val_beat_f1, val_onset_f1, val_tempo_mae = metrics['beat_f1'], metrics['onset_f1'], metrics['tempo_mae']

            val_loss += loss.item() * feats.size(0)
            val_beat_f1s += val_beat_f1 * feats.size(0)
            val_onset_f1s += val_onset_f1 * feats.size(0)
            val_tempo_maes += val_tempo_mae * feats.size(0)

    val_loss /= len(val_ds)
    val_beat_f1s /= len(val_ds)
    val_onset_f1s /= len(val_ds)
    val_tempo_maes /= len(val_ds)
    scheduler.step(val_beat_f1s)  # use onset F1 for scheduler
    if epoch >= 20:
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"LR stepped -- current LR={current_lr:.5f}")

    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} — Train Beat F1: {train_beat_f1s:.4f} — Train Onset F1: {train_onset_f1s:.4f} — Train Tempo MAE: {train_tempo_maes:.4f}")
    print(f"Val Loss: {val_loss:.4f} — Val Beat F1: {val_beat_f1s:.4f} — Val Onset F1: {val_onset_f1s:.4f} — Val Tempo MAE: {val_tempo_maes:.4f}")

    if SEND_TO_WANDB:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_beat_f1": train_beat_f1s,
            "train_onset_f1": train_onset_f1s,
            "train_tempo_mae": train_tempo_maes,
            "val_beat_f1": val_beat_f1s,
            "val_onset_f1": val_onset_f1s,
            "val_tempo_mae": val_tempo_maes,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

    # Early stopping logic
    if epoch >= 20:
         #if val_loss < best_val_loss:
        if val_beat_f1s > best_beat_f1:
            #best_val_loss = val_loss
            best_beat_f1 = val_beat_f1s
            epochs_no_improve = 0
            best_epoch = epoch
            best_ckpt_path = CHECKPOINT_PATH / CHECKPOINT_NAME_STEM.format(epoch=epoch)
            # Save best checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, best_ckpt_path)
            print(f"New best model saved at epoch {epoch} with val_loss {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs (patience={PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val_loss {best_val_loss:.4f}")
            # load best model
            best_ckpt_path = CHECKPOINT_PATH / CHECKPOINT_NAME_STEM.format(epoch=best_epoch)
            checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            # Evaluate on test set
            test_loss = 0.0
            test_beat_f1s = 0.0
            test_onset_f1s = 0.0
            test_tempo_maes = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    feats = batch['features'].to(DEVICE)
                    outs = model(feats)
                    loss, metrics = compute_loss(model, outs, batch, epoch, stage='test')
                    test_beat_f1, test_onset_f1, test_tempo_mae = metrics['beat_f1'], metrics['onset_f1'], metrics['tempo_mae']
                    test_loss += loss.item() * feats.size(0)
                    test_beat_f1s += test_beat_f1 * feats.size(0)
                    test_onset_f1s += test_onset_f1 * feats.size(0)
                    test_tempo_maes += test_tempo_mae * feats.size(0)
            test_loss /= len(test_ds)
            test_beat_f1s /= len(test_ds)
            test_onset_f1s /= len(test_ds)
            test_tempo_maes /= len(test_ds)
            print(
                f"Test Loss: {test_loss:.4f} — Test Beat F1: {test_beat_f1s:.4f} — Test Onset F1: {test_onset_f1s:.4f} — Test Tempo MAE: {test_tempo_maes:.4f}")
            if SEND_TO_WANDB:
                wandb.log({
                    "test_loss": test_loss,
                    "test_beat_f1": test_beat_f1s,
                    "test_onset_f1": test_onset_f1s,
                    "test_tempo_mae": test_tempo_maes,
                    "epoch": epoch,
                })
            break

