from torch import nn
import torch
import torch.nn.functional as F
from models.transformer import Transformer
import math
from torch.autograd import Variable


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=136):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = Variable(self.pe[:x.size(0), :], requires_grad=False)

        x = x + pe
        return self.dropout(x)


def generate_modality_mask():
    """Generate a mask combination that includes at least one modality"""
    combinations = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1]
    ]
    idx = torch.randint(0, 6, [1])
    return combinations[idx[0]]


class ExtremeModalityGenerator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Cross-modal attention enhancement
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
                kdim=hidden_dim,  # Explicitly specify key dimension
                vdim=hidden_dim  # Explicitly specify value dimension
            )
            for _ in range(2)  # Two layers of attention
        ])
        # Conditional gating
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, exist_feats, mask_feat):
        """exist_feats: List of features from existing modalities (e.g., pass [v,f] when generating audio)"""
        # Concatenate existing modality features
        context = torch.cat(exist_feats, dim=1)
        # Multi-level attention
        for attn in self.cross_attn:
            mask_feat, _ = attn(mask_feat, context, context)
        # Gated fusion
        gate = self.gate(torch.cat([mask_feat.mean(1), context.mean(1)], -1))
        return gate.unsqueeze(1) * mask_feat


class MoE_AQA(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder,
                 n_query, dropout, config):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Initialize projection network
        self.in_proj = nn.ModuleDict({
            'v': self._build_proj(in_dim, hidden_dim),
            'a': self._build_proj(in_dim, hidden_dim),
            'f': self._build_proj(1024, hidden_dim)
        })

        # Expert system reconstruction
        self.experts = nn.ModuleDict({
            'v': self._build_expret(hidden_dim, hidden_dim),
            'a': self._build_expret(hidden_dim, hidden_dim),
            'f': self._build_expret(hidden_dim, hidden_dim),
        })

        # Missing modality generation
        self.generators = nn.ModuleDict({
            'v': ExtremeModalityGenerator(self.hidden_dim),
            'a': ExtremeModalityGenerator(self.hidden_dim),
            'f': ExtremeModalityGenerator(self.hidden_dim)
        })

        # Routing network enhancement
        self.router_mix = Mlp(hidden_dim, hidden_dim // 2, 3)

        # Decoding module
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Conv1d(3 * hidden_dim, 2 * hidden_dim, kernel_size=1),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(True),
            nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim)
        )

        self.prototype = nn.Embedding(n_query, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, n_query)
        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).cuda()
        self.pos_encoder = PositionalEncoding(hidden_dim)

    def _build_expret(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim // 2),
            nn.GELU(),
            nn.Conv1d(out_dim // 2, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim)
        )

    def _build_proj(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, in_dim // 2, kernel_size=1),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(in_dim // 2, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, video, audio, flow, mask):
        if self.training:
            return self.forward_train(video, audio, flow, mask)
        else:
            return self.forward_test(video, audio, flow, mask)

    def forward_train(self, video, audio, flow, mask):
        b, t, c = video.shape
        
        # Feature projection
        v = self.in_proj['v'](video.transpose(1, 2)).transpose(1, 2)
        a = self.in_proj['a'](audio.transpose(1, 2)).transpose(1, 2)
        f = self.in_proj['f'](flow.transpose(1, 2)).transpose(1, 2)
        proj_v = self.transformer.encoder(v)
        proj_a = self.transformer.encoder(a)
        proj_f = self.transformer.encoder(f)

        v_mask, a_mask, f_mask = generate_modality_mask(v.size(0), v.device)
        v_mask = proj_v if v_mask == 1 else torch.zeros_like(proj_v)
        a_mask = proj_a if a_mask == 1 else torch.zeros_like(proj_a)
        f_mask = proj_f if f_mask == 1 else torch.zeros_like(proj_f)

        # Detect missing modalities
        missing = []
        if v_mask.sum() == 0: missing.append('v')
        if a_mask.sum() == 0: missing.append('a')
        if f_mask.sum() == 0: missing.append('f')
        # Generate missing modalities
        recon_loss = 0
        for m in missing:
            exist_feats = []
            if 'v' not in missing: exist_feats.append(v_mask)
            if 'a' not in missing: exist_feats.append(a_mask)
            if 'f' not in missing: exist_feats.append(f_mask)

            if m == 'v':
                v_mask = self.generators['v'](exist_feats, torch.zeros_like(v_mask))
                recon_loss += F.mse_loss(v_mask, proj_v)
            elif m == 'a':
                a_mask = self.generators['a'](exist_feats, torch.zeros_like(a_mask))
                recon_loss += F.mse_loss(a_mask, proj_a)
            else:
                f_mask = self.generators['f'](exist_feats, torch.zeros_like(f_mask))
                recon_loss += F.mse_loss(f_mask, proj_f)

        # Generate expert features
        expert_vv, expert_va, expert_vf = self.experts['v'](v_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](v_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](v_mask.transpose(1, 2)).transpose(1, 2)
        expert_av, expert_aa, expert_af = self.experts['v'](a_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](a_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](a_mask.transpose(1, 2)).transpose(1, 2)
        expert_fv, expert_fa, expert_ff = self.experts['v'](f_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](f_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](f_mask.transpose(1, 2)).transpose(1, 2)

        # Calculate routing weights
        router_weight_v = F.softmax(self.router_mix(v_mask), dim=-1)
        router_weight_a = F.softmax(self.router_mix(a_mask), dim=-1)
        router_weight_f = F.softmax(self.router_mix(f_mask), dim=-1)

        # Feature aggregation
        fused_v = torch.cat([expert_vv.unsqueeze(-2), expert_va.unsqueeze(-2), expert_vf.unsqueeze(-2)],dim=-2)
        fusion_v = (fused_v * router_weight_v.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fused_a = torch.cat([expert_av.unsqueeze(-2), expert_aa.unsqueeze(-2), expert_af.unsqueeze(-2)],dim=-2)
        fusion_a = (fused_a * router_weight_a.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fused_f = torch.cat([expert_fv.unsqueeze(-2), expert_fa.unsqueeze(-2), expert_ff.unsqueeze(-2)],dim=-2)
        fusion_f = (fused_f * router_weight_f.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fusion = torch.cat([fusion_v, fusion_a, fusion_f], -1)
        fusion = self.fusion(fusion.transpose(1, 2)).transpose(1, 2)
        # Decoding process
        prototype = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        prototype = self.pos_encoder(prototype)
        fea, att_weights = self.transformer.decoder(prototype, fusion)
        s = self.regressor(fea)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        
        # Generate expert features (for KL loss)
        expert_v = self.experts['v'](proj_v.transpose(1, 2)).transpose(1, 2)
        expert_a = self.experts['a'](proj_a.transpose(1, 2)).transpose(1, 2)
        expert_f = self.experts['f'](proj_f.transpose(1, 2)).transpose(1, 2)
        recon_loss += F.kl_div(torch.log_softmax(fusion_v, -1), torch.softmax(expert_v, -1))
        recon_loss += F.kl_div(torch.log_softmax(fusion_a, -1), torch.softmax(expert_a, -1))
        recon_loss += F.kl_div(torch.log_softmax(fusion_f, -1), torch.softmax(expert_f, -1))
        return {'output': out, 'embed': fea, 'embed2': None, "recon_loss": recon_loss}

    def forward_test(self, video, audio, flow, mask):
        b, t, c = video.shape
        
        # Feature projection
        v_mask, a_mask, f_mask = mask
        if v_mask == 1:
            v = self.in_proj['v'](video.transpose(1, 2)).transpose(1, 2)
            v_mask = self.transformer.encoder(v)
        else:
            v_mask = torch.zeros(b, t, self.hidden_dim).to(video.device)
        if a_mask == 1:
            a = self.in_proj['a'](audio.transpose(1, 2)).transpose(1, 2)
            a_mask = self.transformer.encoder(a)
        else:
            a_mask = torch.zeros(b, t, self.hidden_dim).to(audio.device)
        if f_mask == 1:
            f = self.in_proj['f'](flow.transpose(1, 2)).transpose(1, 2)
            f_mask = self.transformer.encoder(f)
        else:
            f_mask = torch.zeros(b, t, self.hidden_dim).to(flow.device)

        # Detect missing modalities
        missing = []
        if v_mask.sum() == 0: missing.append('v')
        if a_mask.sum() == 0: missing.append('a')
        if f_mask.sum() == 0: missing.append('f')
        # Generate missing modalities
        for m in missing:
            exist_feats = []
            if 'v' not in missing: exist_feats.append(v_mask)
            if 'a' not in missing: exist_feats.append(a_mask)
            if 'f' not in missing: exist_feats.append(f_mask)

            if m == 'v':
                v_mask = self.generators['v'](exist_feats, torch.zeros_like(v_mask))
            elif m == 'a':
                a_mask = self.generators['a'](exist_feats, torch.zeros_like(a_mask))
            else:
                f_mask = self.generators['f'](exist_feats, torch.zeros_like(f_mask))

        # Generate expert features
        expert_vv, expert_va, expert_vf = self.experts['v'](v_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](v_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](v_mask.transpose(1, 2)).transpose(1, 2)
        expert_av, expert_aa, expert_af = self.experts['v'](a_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](a_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](a_mask.transpose(1, 2)).transpose(1, 2)
        expert_fv, expert_fa, expert_ff = self.experts['v'](f_mask.transpose(1, 2)).transpose(1, 2), self.experts['a'](f_mask.transpose(1, 2)).transpose(1, 2), self.experts['f'](f_mask.transpose(1, 2)).transpose(1, 2)

        # Calculate routing weights
        router_weight_v = F.softmax(self.router_mix(v_mask), dim=-1)
        router_weight_a = F.softmax(self.router_mix(a_mask), dim=-1)
        router_weight_f = F.softmax(self.router_mix(f_mask), dim=-1)

        # Feature aggregation
        fused_v = torch.cat([expert_vv.unsqueeze(-2), expert_va.unsqueeze(-2), expert_vf.unsqueeze(-2)],dim=-2)
        fusion_v = (fused_v * router_weight_v.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fused_a = torch.cat([expert_av.unsqueeze(-2), expert_aa.unsqueeze(-2), expert_af.unsqueeze(-2)],dim=-2)
        fusion_a = (fused_a * router_weight_a.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fused_f = torch.cat([expert_fv.unsqueeze(-2), expert_fa.unsqueeze(-2), expert_ff.unsqueeze(-2)],dim=-2)
        fusion_f = (fused_f * router_weight_f.unsqueeze(-1).repeat(1, 1, 1, self.hidden_dim)).sum(dim=2)
        fusion = torch.cat([fusion_v, fusion_a, fusion_f], -1)
        fusion = self.fusion(fusion.transpose(1, 2)).transpose(1, 2)
        # Decoding process
        prototype = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        prototype = self.pos_encoder(prototype)
        fea, att_weights = self.transformer.decoder(prototype, fusion)
        s = self.regressor(fea)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)
        return {'output': out}
