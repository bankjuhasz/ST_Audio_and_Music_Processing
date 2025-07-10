from collections import OrderedDict
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import math


class ThreeHeadedDragon(nn.Module):
    '''
    A 3-headed Dragon to handle onset, beat, and tempo detection with one head for each task!
    The body of the dragon is - of course - shared between the three heads. How else could it be???

    The preprocessed spectrograms go into the dragon's stomach (FrontendStomach), but they need to go through the rest
    of body (TCN block) to be ready for the heads(?) and their fiery predictions.

    The metaphor at this point is a bit tortured - this isn't a gastroenterological model, after all.
    '''
    def __init__(
            self,
            freq_dim,
            stomach_dim,
    ):
        super(ThreeHeadedDragon, self).__init__()

        ### MODEL PARAMS ###

        # add one log‐variance per task --> attempting to learn the uncertainty of each task
        # initialize to zero so initial weight = 1/(2*exp(0)) = 0.5
        self.raw_log_var_on = nn.Parameter(torch.zeros(()))
        self.raw_log_var_be = nn.Parameter(torch.zeros(()))
        self.raw_log_var_tm = nn.Parameter(torch.zeros(()))

        # creating the stomach
        self.frontendstomach = FrontEndStomach()

        # body (i.e., TCN blocks)
        self.tcn = TCNWithAccum([TCNBlock(224, kernel_size=5, dilation=2 ** d, dropout=0.1) for d in range(11)])

        # heads
        self.heads = self.make_heads(128, freq_dim) # using the last body output channels

        # prepare a dict to hold activations
        self._acts = {}

        # register hooks on some layers --> for debugging
        for name, module in self.frontendstomach.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
                module.register_forward_hook(self._make_hook(f"stomach.{name}"))

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = self.frontendstomach(x)
        #print(f"Shape after frontendstomach: {x.shape}")
        #print(f"Shape before tcn: {x.shape}")
        x, tempo_feat = self.tcn(x)
        #print(f"Shape after tcn: {x.shape}")
        return {
            "onset": self.heads["onset"](x),
            "beat": self.heads["beat"](x),
            "tempo": self.heads["tempo"](tempo_feat),
        }

    @staticmethod
    def make_heads(channels, freq_dim):
        return nn.ModuleDict({
            "onset": nn.Sequential(
                Rearrange("b c t -> b t c"),  # [B,channels,T] -> [B,T,channels]
                nn.Linear(224, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                #nn.Dropout(0.1),
                nn.Linear(128, 1),  # -> [B,T,512]
                #nn.Sigmoid(),
            ),
            "beat": nn.Sequential(
                Rearrange("b c t -> b t c"),  # [B,channels,T] -> [B,T,channels]
                nn.Linear(224, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                #nn.Dropout(0.1),
                nn.Linear(128, 1),  # -> [B,T,512]
                #nn.Sigmoid(),
            ),
            "tempo": nn.Sequential(
                #Rearrange("b t d -> b d t"),
                nn.AdaptiveAvgPool1d(1), # [B, C*F, T] -> [B,C*F,1]
                #nn.Flatten(1), # [B,C*F,1] -> [B,C*F]
                Rearrange("b c 1 -> b c"),  # [B,C*F, 1] -> [B,C*F]
                nn.Linear(224, 300)
            )
        })

    def _make_hook(self, name):
        def hook(module, inp, out):
            # store a detached copy so it doesn’t hold the graph
            self._acts[name] = out.detach()
        return hook


class TCNWithAccum(nn.Module):
    """ TCN which is able to accumulate intermediate outputs and return them. """
    def __init__(self, tcn_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(tcn_blocks)

    def forward(self, x):
        accum = 0
        for block in self.blocks:
            y, x = block(x, return_y=True)
            accum = accum + y
        return x, accum

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

    def forward(self, x, return_y=False):
        y = self.conv2(self.drop(self.elu(self.conv(x))))
        r = self.res_conv(x)
        out = r + y # residual connection
        if return_y:
            return y, out # return both the output and the intermediate y for accumulation for tempo predictions
        # if not returning y, just return the output --> original TCN behavior
        return out


class FrontEndStomach(nn.Module):
    """
    2 conv+pool blocks followed by 1 conv block
    """
    def __init__(self,
                 in_mels: int = 128,
                 n_filters: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            # block 1
            nn.Conv2d(1, n_filters, kernel_size=(3,3), padding=(1,1), bias=False),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)), # pool only in freq, no overlaps
            nn.Dropout2d(dropout),

            # block 2
            nn.Conv2d(n_filters, n_filters, (3,3), padding=(1,1), bias=False),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Dropout2d(dropout),

            # block 3
            nn.Conv2d(n_filters, n_filters, (1,5), padding='same', bias=False),
            nn.GELU(),
            #nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),
            nn.Dropout2d(dropout),

            Rearrange("b c f t -> b (c f) t"),
        )

    def forward(self, x):
        return self.net(x)
