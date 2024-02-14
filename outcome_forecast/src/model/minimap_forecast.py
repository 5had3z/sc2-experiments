from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from konductor.data import get_dataset_properties
from konductor.init import ModuleInitConfig
from konductor.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig


@MODEL_REGISTRY.register_module("temporal-conv")
class TemporalConv(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        out_ch: int,
        n_timesteps: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_ch, hidden_ch, (n_timesteps // 2 + 1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(hidden_ch),
            activation(),
            nn.Conv3d(hidden_ch, out_ch, (n_timesteps // 2, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_ch),
            activation(),
        )
        self.out_ch = out_ch


@dataclass
class BaseConfig(TorchModelConfig):
    encoder: ModuleInitConfig
    temporal: ModuleInitConfig
    decoder: ModuleInitConfig
    history_len: int = 8

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args
        model_cfg["encoder"]["args"]["in_ch"] = props["image_ch"]
        return super().from_config(config, idx)

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            self.encoder = ModuleInitConfig(**self.encoder)
        if isinstance(self.temporal, dict):
            self.temporal = ModuleInitConfig(**self.temporal)
        if isinstance(self.decoder, dict):
            self.decoder = ModuleInitConfig(**self.decoder)


@dataclass
@MODEL_REGISTRY.register_module("conv-forecast-v1")
class ConvV1Config(BaseConfig):
    def __post_init__(self):
        super().__post_init__()
        assert self.history_len % 2 == 0, "history_len must be even"
        self.temporal.args["n_timesteps"] = self.history_len

    def get_instance(self, *args, **kwargs) -> Any:
        """Construct modules and return conv forecaster"""
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)

        self.temporal.args["in_ch"] = encoder.out_ch[-1]
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.decoder.args["in_ch"] = temporal.out_ch + encoder.out_ch[0]
        decoder = MODEL_REGISTRY[self.decoder.type](**self.decoder.args)
        return ConvForecaster(encoder, temporal, decoder, self.history_len)


class ConvForecaster(nn.Module):
    """
    Use Siamese ConvNet to Extract features
    3dConv Along Features
    Upsample and Concatenate with Last Image
    Conv Decode TemporalFeatures+Last Image for Final Output
    """

    is_logit_output = True

    def __init__(
        self,
        encoder: nn.Module,
        temporal: nn.Module,
        decoder: nn.Module,
        history_len: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.temporal_conv = temporal
        self.decoder = decoder
        self.history_len = history_len

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        minimap_low: list[Tensor] = []
        minimap_high: list[Tensor] = []

        for t in range(inputs.shape[1]):
            enc = self.encoder(inputs[:, t])
            minimap_low.append(enc[0])
            minimap_high.append(enc[-1])

        stacked_feats = torch.stack(minimap_low, dim=2)
        temporal_feats = self.temporal_conv(stacked_feats)
        temporal_feats = F.interpolate(
            temporal_feats.squeeze(2),
            size=minimap_high[-1].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        cat_features = torch.cat([temporal_feats, minimap_high[-1]], dim=1)
        decoded = self.decoder(cat_features)
        return decoded

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """"""
        minimaps = inputs["minimap_features"]
        ntime = minimaps.shape[1]
        preds: list[Tensor] = []
        for start_idx in range(ntime - self.history_len):
            end_idx = start_idx + self.history_len
            pred = self.forward_sequence(minimaps[:, start_idx:end_idx])
            pred = F.interpolate(
                pred, mode="bilinear", size=minimaps.shape[-2:], align_corners=True
            )
            preds.append(pred)

        out = torch.stack(preds, dim=1)
        return out


@MODEL_REGISTRY.register_module("cross-attn-block-v1")
class CrossAttentionBlockV1(nn.Module):
    """CrossAttention -> SelfAttention Module"""

    def __init__(self, q_ch: int, kv_ch: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.q_norm = nn.LayerNorm(q_ch)
        self.kv_norm = nn.LayerNorm(kv_ch)
        self.c_attn = nn.MultiheadAttention(
            embed_dim=q_ch,
            kdim=kv_ch,
            vdim=kv_ch,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.middle_mlp = CrossAttentionBlockV1.make_mlp(q_ch)
        self.middle_norm = nn.LayerNorm(q_ch)
        self.s_attn = nn.MultiheadAttention(
            embed_dim=q_ch, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.out_mlp = CrossAttentionBlockV1.make_mlp(q_ch)

    @staticmethod
    def make_mlp(ch: int):
        return nn.Sequential(
            nn.LayerNorm(ch), nn.Linear(ch, ch), nn.GELU(), nn.Linear(ch, ch)
        )

    def forward(self, query: Tensor, keyvalue: Tensor):
        inter: Tensor
        inter = query + self.c_attn(self.q_norm(query), self.kv_norm(keyvalue))
        inter = inter + self.middle_mlp(inter)
        inter = inter + self.s_attn(self.middle_norm(inter))
        inter = inter + self.out_mlp(inter)
        return inter


@MODEL_REGISTRY.register_module("transformer-resampler")
class TransformerResampler(nn.Module):
    """
    Alternate between cross-attn on input data and
    self-attn over latent to decode.
    """

    def __init__(
        self, latent_dim: int, kv_dim: int, num_blocks: int, num_heads: int = 4
    ) -> None:
        super().__init__()
        self.cross_attn_blocks = nn.ModuleList(
            CrossAttentionBlockV1(latent_dim, kv_dim, num_heads)
            for _ in range(num_blocks)
        )

    def forward(self, query: Tensor, keyvalue: Tensor):
        for module in self.cross_attn_blocks:
            query = module(query, keyvalue)
        return query


class PosQueryDecoder(nn.Module):
    def __init__(
        self,
        out_shape: tuple[int, int],
        input_dim: int,
        output_dim: int,
        num_heads: int,
        query_cfg: ModuleInitConfig,
    ) -> None:
        super().__init__()

        self.out_shape = out_shape
        self.queries: nn.Parameter | Tensor = {
            "fixed-queries": make_fixed_queries,
            "learned-queries": make_learned_queries,
            "learned-queries-freq": make_learned_freq_queries,
        }[query_cfg.type](out_shape=out_shape, **query_cfg.args)

        self.input_norm = nn.LayerNorm(input_dim)
        self.decoder = nn.MultiheadAttention(
            embed_dim=self.queries.shape[-1],
            kdim=input_dim,
            vdim=input_dim,
            num_heads=num_heads,
        )
        self.linear = nn.Linear(self.queries.shape[-1], output_dim)

    def forward(self, latent: Tensor):
        queries = self.queries.repeat(latent.shape[0], 1, 1)
        decoded: Tensor = self.decoder(queries, self.input_norm(latent))
        decoded = self.linear(decoded)
        decoded = decoded.reshape(latent.shape[0], -1, *self.out_shape)
        return decoded


def make_learned_queries(out_shape: tuple[int, int], query_dim: int):
    """Make parameters for learned queries"""
    queries = nn.Parameter(torch.empty(out_shape[0] * out_shape[1], query_dim))
    with torch.no_grad():
        queries.normal_(0, 0.5).clamp_(-2, 2)
    return queries


def make_fixed_queries(
    out_shape: tuple[int, int], f_num: int, f_max: float | None = None
):
    """Make fixed queries with sine/cosine encodings"""
    if f_max is None:
        f_max = float(max(out_shape))

    coords = torch.stack(
        torch.meshgrid(
            [torch.linspace(-1, 1, steps=s) for s in out_shape], indexing="ij"
        ),
        dim=-1,
    )
    frequencies = torch.linspace(1.0, f_max / 2.0, f_num)[None, None, None]
    frequency_grids = torch.pi * coords[..., None] * frequencies
    encodings = torch.cat([frequency_grids.sin(), frequency_grids.cos()], dim=-1)

    return encodings.reshape(*out_shape, -1)


def make_learned_freq_queries(
    out_shape: tuple[int, int], f_num: int, f_max: float | None = None
):
    """Initialize parameter with sine/cosine encodings"""
    queries = make_fixed_queries(out_shape, f_num, f_max)
    return nn.Parameter(queries)


class TransformerForecasterV1(nn.Module):
    """Just do the ol' flatten, and MHA over time and space"""

    is_logit_output = True

    def __init__(
        self,
        encoder: nn.Module,
        temporal: nn.Module,
        decoder: nn.Module,
        num_latents: int,
        latent_dim: int,
        history_len: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.temporal = temporal
        self.decoder = decoder
        self.history_len = history_len
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

    @torch.no_grad()
    def _init_parameters(self):
        self.latent.normal_(0, 0.5).clamp_(-2.0, 2.0)

    def flatten_input_encodings(self, inputs: Tensor):
        """Transform inputs [B,T,C,H,W] to [B,THW,C] and add position embeddings"""
        return inputs

    def forward_sequence(self, inputs: Tensor):
        """Run a single sequence [B,T,C,H,W]"""
        inputs_enc = torch.stack(
            [self.encoder(inputs[:, t]) for t in range(inputs.shape[1])], dim=1
        )
        inputs_enc = self.flatten_input_encodings(inputs_enc)
        temporal_feats = self.temporal(self.latent, inputs_enc)
        output = self.decoder(temporal_feats)
        return output

    def forward(self, inputs: dict[str, Tensor]):
        """If input sequence is longer than designated, 'convolve' over input"""
        minimaps = inputs["minimap_features"]
        ntime = minimaps.shape[1]
        preds: list[Tensor] = []
        for start_idx in range(ntime - self.history_len):
            end_idx = start_idx + self.history_len
            pred = self.forward_sequence(minimaps[:, start_idx:end_idx])
            preds.append(pred)

        out = torch.stack(preds, dim=1)
        return out


@dataclass
@MODEL_REGISTRY.register_module("transformer-forecast-v1")
class TransformerForecasterConfig(BaseConfig):
    decoder_query: ModuleInitConfig = field(kw_only=True)
    num_latents: int = field(kw_only=True)
    latent_dim: int = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.decoder_query, ModuleInitConfig):
            self.decoder_query = ModuleInitConfig(**self.decoder_query)

    def get_instance(self, *args, **kwargs) -> Any:
        encoder = MODEL_REGISTRY[self.encoder.type](**self.encoder.args)
        if hasattr(encoder, "disable_fpn"):
            assert encoder.disable_fpn
        self.temporal.args["latent_dim"] = self.latent_dim
        self.temporal.args["kv_dim"] = encoder.out_ch
        temporal = MODEL_REGISTRY[self.temporal.type](**self.temporal.args)

        self.decoder.args["input_dim"] = self.latent_dim
        decoder = PosQueryDecoder(query_cfg=self.decoder_query, **self.decoder.args)

        return self.init_auto_filter(
            TransformerForecasterV1, encoder=encoder, temporal=temporal, decoder=decoder
        )
