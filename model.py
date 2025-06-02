from collections import OrderedDict
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import math


class ThreeHeadedDragon(nn.Module):
    '''
    A 3-headed Dragon to handle onset, beat, and tempo detection with one head for each task!
    The body of the dragon is - of course - shared between the three heads. How else could it be???

    The preprocessed spectrograms are already in the dragon's stomach, but they need to go through the body to be ready
    for the heads(?) and their fiery predictions.

    The metaphor at this point is a bit tortured - this isn't a gastroenterological model, after all.
    '''
    def __init__(
            self,
            freq_dim,
            stomach_dim,
    ):
        super(ThreeHeadedDragon, self).__init__()

        ### MODEL PARAMS ###

        # Add one log‐variance per task --> attempting to learn the uncertainty of each task
        # initialize to zero so initial weight = 1/(2*exp(0)) = 0.5
        self.raw_log_var_on = nn.Parameter(torch.zeros(()))
        self.raw_log_var_be = nn.Parameter(torch.zeros(()))
        self.raw_log_var_tm = nn.Parameter(torch.zeros(()))


        ### STOMACH AND BODY ###

        # creating the stomach
        '''self.stomach = self.make_stomach(freq_dim, stomach_dim)'''
        self.frontendstomach = FrontEndStomach()

        #self.avgpool = nn.AdaptiveAvgPool2d((1, None))  # reduce frequency dimension to 1, keep time dimension
        #self.rearrange_before_tcn = Rearrange("b c 1 t -> b c t")

        # creating the body
        '''body_blocks = []
        channels = stomach_dim
        for _ in range(3):
            body_blocks.append(self.make_body(in_channels=channels, out_channels=channels*2))
            channels *= 2
        freq_dim = 4
        self.body = nn.Sequential(*body_blocks)'''


        ### TCN BLOCKS ###
        tcn_layers = []
        for i, d in enumerate(range(0, 11)):
            tcn_layers.append(TCNBlock(224, kernel_size=5, dilation=2 ** d, dropout=0.1))
        self.tcn = nn.Sequential(*tcn_layers)


        ### POSITIONAL ENCODING FOR TRANSFORMER ###
        '''
        d_model = channels * freq_dim
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape [max_seq_len, d_model]
        self.register_buffer('pos_enc', pe)'''


        ### TRANSFORMER ###

        # magic (transformer)
        '''self.rearrange_for_transformer = Rearrange("b c f t -> b t (c f)")

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True),
            num_layers=2
        )'''

        ### HEADS ###
        self.heads = self.make_heads(128, freq_dim) # using the last body output channels

        # prepare a dict to hold activations
        self._acts = {}

        # register hooks on some layers
        for name, module in self.frontendstomach.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
                module.register_forward_hook(self._make_hook(f"stomach.{name}"))

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = self.frontendstomach(x)
        #print(f"Shape after frontendstomach: {x.shape}")
        #x = self.avgpool(x)
        #x = self.rearrange_before_tcn(x)
        #x = self.body(x)
        #B,C,F,T = x.shape  # B: batch size, C: channels, F: frequency bins, T: time frames
        #x = x.view(B, C * F, T)
        #print(f"Shape before tcn: {x.shape}")
        x = self.tcn(x)
        #print(f"Shape after tcn: {x.shape}")
        #x = self.rearrange_for_transformer(x)  # [B, T, C*F]
        #print(f"Shape after rearrange: {x.shape}")
        #T = x.size(1)
        #x = x + self.pos_enc[:T, :].unsqueeze(0)  # [1, T, D] -> [B, T, D]
        #x = self.transformer(x)  # [B, T, C*F]
        #print(f"Shape before heads: {x.shape}")
        return {
            "onset": self.heads["onset"](x),
            "beat": self.heads["beat"](x),
            "tempo": self.heads["tempo"](x)
        }

    @staticmethod
    def make_stomach(freq_dim, stomach_dim):
        conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=stomach_dim,
            kernel_size=(4, 3),
            stride=(4, 1),
            padding=(0, 1),
            bias=False
        )
        stomach = nn.Sequential(
            OrderedDict(
                rearrange_tf = Rearrange('b 1 f t -> b f t'),
                batch_norm_1d = nn.BatchNorm1d(num_features=freq_dim), # normalize each mel-bin independently (treat freq_dim as channels temporarily)
                rearrange_add_channel=Rearrange("b f t -> b 1 f t"), # add back channel dimension
                conv_2d = conv2d,
                bn2d=nn.BatchNorm2d(stomach_dim), # normalize across channels
                activation=nn.GELU(),
            )
        )
        return stomach

    @staticmethod
    def make_body(in_channels, out_channels):
        conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2, 3),
            stride=(2, 1),
            padding=(0, 1),
            bias=False,
        )
        body = nn.Sequential(
            OrderedDict(
                conv_2d=conv2d,
                batch_norm_2d=nn.BatchNorm2d(out_channels),
                activation=nn.ReLU(inplace=True),
                dropout=nn.Dropout(0.1),
            )
        )
        return body

    @staticmethod
    def make_heads(channels, freq_dim):
        return nn.ModuleDict({
            "onset": nn.Sequential(
                Rearrange("b c t -> b t c"),  # [B,channels,T] → [B,T,channels]
                nn.Linear(224, 128),
                nn.GELU(),
                nn.Linear(128, 1),  # → [B,T,512]
                #nn.Sigmoid(),
            ),
            "beat": nn.Sequential(
                Rearrange("b c t -> b t c"),  # [B,channels,T] → [B,T,channels]
                nn.Linear(224, 128),
                nn.GELU(),
                nn.Linear(128, 1),  # → [B,T,512]
                #nn.Sigmoid(),
            ),
            "tempo": nn.Sequential(
                #Rearrange("b t d -> b d t"),
                nn.AdaptiveAvgPool1d(1), # [B, C*F, T] → [B, C*F, 1]
                nn.Flatten(1), # [B, C*F, 1] → [B, C*F]
                nn.Linear(224, 3)
            )
        })

    def _make_hook(self, name):
        def hook(module, inp, out):
            # store a detached copy so it doesn’t hold the graph
            self._acts[name] = out.detach()
        return hook

    '''
    "onset": nn.Sequential(
        nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True),
        nn.Conv2d(channels, 1, 1),
        # average out frequency dimension, keep time dimension:
        nn.AvgPool2d(kernel_size=(4, 1)),
        Rearrange("b 1 1 t -> b t 1"),
        #nn.Sigmoid(),
    ),
    "beat": nn.Sequential(
        nn.Conv2d(channels, channels, 1), nn.ReLU(inplace=True),
        nn.Conv2d(channels, 1, 1),
        nn.AvgPool2d(kernel_size=(4, 1)),
        Rearrange("b 1 1 t -> b t 1"),
        #nn.Sigmoid(),
    ),
    '''

class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) * dilation // 2
        # current signal path
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.elu = nn.ELU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # residual connection
        self.res_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv2(self.drop(self.elu(self.conv(x))))
        #y = self.conv2(self.elu(self.conv(x)))
        r = self.res_conv(x)

        # residual connection
        #print(f"Input shape: {x.shape}, Conv output shape: {y.shape}, Residual shape: {r.shape}")
        return r + y

class FrontEndStomach(nn.Module):
    """
    3 conv+pool blocks:
      1) Conv2d(1 → 16, k=(3×3)) → ELU → MaxPool(freq,3) → Dropout(0.1)
      2) Conv2d(16→16, k=(3×3)) → ELU → MaxPool(freq,3) → Dropout(0.1)
      3) Conv2d(16→16, k=(1×8)) → ELU → MaxPool(freq,3) → Dropout(0.1)
    Resulting feature per time-frame: 16-dim
    """
    def __init__(self,
                 in_mels: int = 128,
                 n_filters: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            # --- block 1 ---
            nn.Conv2d(1, n_filters, kernel_size=(3,3), padding=(1,1), bias=False),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)), # pool only in freq, no overlaps
            nn.Dropout2d(dropout),

            # --- block 2 ---
            nn.Conv2d(n_filters, n_filters, (3,3), padding=(1,1), bias=False),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Dropout2d(dropout),

            # --- block 3 ---
            nn.Conv2d(n_filters, n_filters, (1,8), padding='same', bias=False),
            nn.GELU(),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Dropout2d(dropout),

            # now shape is [B, 16, F_out, T]
            # we want [B, T, 16] → TCN expects (batch, time, features)
            #nn.AdaptiveAvgPool2d((1, None)), # reduce frequency dimension to 1, keep time dimension
            #Rearrange("b c 1 t -> b c t"),  # [B, C, 1, T] → [B, C, T]
            Rearrange("b c f t -> b (c f) t"),
        )

    def forward(self, x):
        """
        x: [B, 1, F, T]
        returns: [B, T, 16]
        """
        return self.net(x)
