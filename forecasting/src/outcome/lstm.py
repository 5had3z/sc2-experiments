import torch
from konductor.models import MODEL_REGISTRY
from torch import Tensor, nn


@MODEL_REGISTRY.register_module("lstm-v1")
class LSTMDecoderV1(nn.Module):
    is_logit_output = True

    def __init__(
        self,
        in_ch: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        residule_add: bool = False,
    ):
        super().__init__()
        self.latent = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.h0 = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
        self.c0 = nn.Parameter(torch.empty(num_layers, 1, hidden_size))
        torch.nn.init.xavier_normal_(self.h0)
        torch.nn.init.xavier_normal_(self.c0)

        self.residule_add = residule_add
        if residule_add:
            self.residule_squeeze = nn.Linear(in_ch, hidden_size)

        self.decode = nn.Linear(hidden_size, 1)

    def forward(self, inputs: Tensor):
        """Input is a [B, T, C] tensor, returns [B, T]"""
        batch_sz = inputs.shape[0]
        h, c = self.h0.repeat(1, batch_sz, 1), self.c0.repeat(1, batch_sz, 1)

        out, _ = self.latent(inputs, (h, c))
        if self.residule_add:
            out = out + self.residule_squeeze(inputs)
        outputs: Tensor = self.decode(out)
        return outputs.squeeze(-1)
