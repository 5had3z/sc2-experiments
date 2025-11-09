import torch
from konductor.models import MODEL_REGISTRY
from torch import Tensor, nn


@MODEL_REGISTRY.register_module("transformer-v1")
class TransformerDecoderV1(nn.Module):
    is_logit_output = True

    def __init__(
        self,
        in_ch: int,
        max_time: float,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_head: int = 4,
        num_time_freq: int = 8,
        max_time_freq: int = 8,
        residule_add: bool = False,
    ):
        """Max time in minutes, will be converted internally to gameloops"""
        super().__init__()
        self.squeeze = nn.Linear(in_ch, hidden_size)
        self.decode = nn.Linear(hidden_size + num_time_freq * 2, 1)

        self.max_time = max_time / (22.4 * 60)
        self.num_freq = num_time_freq
        self.max_freq = max_time_freq

        enc_layer = nn.TransformerEncoderLayer(
            hidden_size + self.num_freq * 2, nhead=num_head
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=num_layers,
        )

        self.mask: Tensor
        self.register_buffer("mask", torch.empty([1]), persistent=False)
        self.residule_add = residule_add
        if residule_add:
            self.residule_linear = nn.Linear(in_ch, hidden_size + self.num_freq * 2)

    @torch.no_grad()
    def create_time_embeddings(self, timepoints: Tensor):
        """Create position embeddings based on timepoints relative to maximum time"""
        frequencies = torch.linspace(
            1.0, self.max_freq / 2.0, self.num_freq, device=timepoints.device
        )
        timepoints_norm = timepoints / self.max_time
        frequency_grid = timepoints_norm[..., None] * frequencies[None, ...]
        encodings = [
            torch.sin(torch.pi * frequency_grid),
            torch.cos(torch.pi * frequency_grid),
        ]
        return torch.cat(encodings, dim=-1)

    def get_mask(self, time_dim: int) -> Tensor:
        if time_dim != self.mask.shape[0]:
            self.mask = nn.Transformer.generate_square_subsequent_mask(
                time_dim, device=self.mask.device
            )
        return self.mask

    def forward(self, inputs: Tensor, timesteps: Tensor):
        """Input is a [B, T, C] tensor, timesteps is [B, T] in gamestep units, returns [B, T]"""
        time_embed = self.create_time_embeddings(timesteps)
        cat_features = torch.cat([self.squeeze(inputs), time_embed], dim=-1)

        temporal_enc: Tensor = self.temporal_encoder(
            cat_features, mask=self.get_mask(cat_features.shape[1]), is_causal=True
        )

        if self.residule_add:
            temporal_enc = temporal_enc + self.residule_linear(inputs)

        decoded: Tensor = self.decode(temporal_enc)
        return decoded.squeeze(-1)
