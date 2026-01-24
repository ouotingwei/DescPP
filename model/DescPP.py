import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def MLP(channels: List[int], do_bn: bool=False) -> nn.Module:
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Linear(channels[i-1], channels[i]))
        if i < (len(channels)-1):
            if do_bn: layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class LFFKeypointEncoder(nn.Module):
    def __init__(self, in_dims: list, f_dims: list, gammas: list, mlp_layers: list, out_d: int):
        super().__init__()
        self.in_dims = in_dims
        self.f_dims = f_dims
        
        self.Wr_list = nn.ParameterList()
        for f_dim, gamma, in_dim in zip(f_dims, gammas, in_dims):
            W = torch.empty(f_dim // 2, in_dim)
            nn.init.normal_(W, mean=0, std=gamma**-2)
            self.Wr_list.append(nn.Parameter(W))

        total_f_dim = sum(f_dims)
        self.mlp = MLP([total_f_dim] + mlp_layers + [out_d])

    def forward(self, x: torch.Tensor):
        proj_list = []
        start = 0
        for i, dim in enumerate(self.in_dims):
            feat = x[:, start : start + dim]
            start += dim
            proj = feat @ self.Wr_list[i].T
            F_feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / np.sqrt(self.f_dims[i])
            proj_list.append(F_feat)

        return self.mlp(torch.cat(proj_list, dim=-1))

class DescriptorEncoder(nn.Module):
    def __init__(self, descriptor_dim: int, layers: List[int], dropout: bool=False, p: float=0.1) -> None:
        super().__init__()
        self.encoder = MLP([descriptor_dim] + layers + [descriptor_dim])
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, descs):
        residual = descs
        if self.use_dropout:
            return residual + self.dropout(self.encoder(descs))
        return residual + self.encoder(descs)
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, descriptor_dim: int, dropout:bool=False, p: float=0.1) -> None:
        super().__init__()
        self.mlp = MLP([descriptor_dim, descriptor_dim*2, descriptor_dim])
        self.layer_norm = nn.LayerNorm(descriptor_dim, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class AFTAttention(nn.Module):
    """ Attention-free attention """
    def __init__(self, d_model: int, dropout: bool = False, p: float = 0.1) -> None:
        super().__init__()
        self.dim = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = torch.sigmoid(q)
        k = k.transpose(-1, -2)
        k = torch.softmax(k, dim=-1)
        k = k.transpose(-1, -2)
        kv = (k * v).sum(dim=-2, keepdim=True)
        x = q * kv
        x = self.proj(x)
        if self.use_dropout:
            x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class MambaAFTsMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        # mamba
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # aft-s
        self.aft = AFTAttention(d_model=self.d_model)
        # fuse
        self.fuse_score = nn.Linear(self.d_model * 2, 2)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape

        conv_state, ssm_state = None, None
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None:  # Doesn't support outputting the states
            mamba_out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            mamba_out = self.out_proj(y)
                
        # AFT branch
        aft_out = self.aft(hidden_states)

        # fusion gate
        score = self.fuse_score(torch.cat([aft_out, mamba_out], dim=-1))
        gate = torch.softmax(score, dim=-1) 

        aft_w = gate[..., 0].unsqueeze(-1)
        mamba_w = gate[..., 1].unsqueeze(-1)
        out = aft_w * aft_out + mamba_w * mamba_out

        return out
    
class MambaAFTsLayer(nn.Module):
    def __init__(self, descriptor_dim: int, d_state: int = 16, d_conv: int = 3, expand: int = 2,
                 dropout: bool=False, p: float=0.1):
        super().__init__()
        self.use_dropout = dropout
        self.dropout = nn.Dropout(p=p) if dropout else nn.Identity()

        self.mamba_aft_mixer = MambaAFTsMixer(
            d_model=descriptor_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm1 = nn.LayerNorm(descriptor_dim, eps=1e-6)
        self.ffn = PositionwiseFeedForward(descriptor_dim, dropout=dropout, p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_back = False
        if x.dim() == 2:
            # [N, D] -> [1, N, D]
            x = x.unsqueeze(0)
            squeeze_back = True
        assert x.dim() == 3

        residual = x
        y = self.mamba_aft_mixer(x) # [B, N, D]
        y = self.dropout(y)
        y = self.norm1(y + residual) # residual+LN
        y = self.ffn(y)               

        if squeeze_back:
            y = y.squeeze(0) # [N, D]
        return y

class MambaAFTsNN(nn.Module):
    def __init__(self, descriptor_dim: int, layer_num: int,
                 d_state: int = 16, d_conv: int = 3, expand: int = 2,
                 dropout: bool=False, p: float=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MambaAFTsLayer(descriptor_dim, d_state=d_state, d_conv=d_conv, expand=expand,
                       dropout=dropout, p=p)
            for _ in range(layer_num)
        ])

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        x = desc
        for layer in self.layers:
            x = layer(x)
        return x

class DescPP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        d_dim = config['descriptor_dim']
        dropout_p = config.get('dropout_p', 0.1)

        # Descriptor Encoder
        self.denc = DescriptorEncoder(
            descriptor_dim=d_dim, 
            layers=config['denc_mlp_layers'], 
            dropout=True, p=dropout_p
        )

        # Keypoint Encoder
        self.use_kenc = config.get('use_kenc', True)
        if self.use_kenc:
            self.kenc = LFFKeypointEncoder(
                in_dims=config['kenc_in_dims'],
                f_dims=config['kenc_f_dims'],
                gammas=config['kenc_gammas'],
                mlp_layers=config['kenc_mlp_layers'],
                out_d=d_dim
            )

        # Mamba-AFT Mixer Layers
        self.use_cross = config.get('use_cross', True)
        if self.use_cross:
            self.layer_norm = nn.LayerNorm(d_dim, eps=1e-6)
            self.attn_proj = MambaAFTsNN(
                descriptor_dim=d_dim,
                layer_num=config['mamba_layers'],
                d_state=config['d_state'],
                d_conv=config['d_conv'],
                expand=config['expand'],
                dropout=True, p=dropout_p
            )

        # Final Output 
        self.final_proj = nn.Linear(d_dim, d_dim)
        
        # activation
        act_type = str(config.get('final_activation', 'none')).lower()
        self.last_activation = nn.Tanh() if act_type == 'tanh' else nn.Identity()
            
        # normalization
        self.use_normalize = config.get('use_normalize', False)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, desc, kpts):
        # Local Fusion
        x = self.denc(desc)
        if self.use_kenc:
            x = x + self.kenc(kpts)
            x = self.dropout(x)
        # Context Aggregation
        if self.use_cross:
            x = self.attn_proj(self.layer_norm(x))
        
        x = self.final_proj(x)
        x = self.last_activation(x)
        
        if self.use_normalize:
            x = F.normalize(x, dim=-1)
        return x