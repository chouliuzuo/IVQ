"""Vector quantizer.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: 
https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
https://github.com/google-research/magvit/blob/main/videogvt/models/vqvae.py
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/distributions/distributions.py
https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py
"""
from typing import Mapping, Text, Tuple, List

import torch
from einops import rearrange
from accelerate.utils.operations import gather
from torch.cuda.amp import autocast
import torch.nn.functional as F

gua_codes = [
[1,1,1,1,1,1], [0,0,0,0,0,0], [1,0,0,0,1,0], [0,1,0,0,0,1],
[1,1,1,0,1,0], [0,1,0,1,1,1], [0,1,0,0,0,0], [0,0,0,0,1,0],
[1,1,1,0,1,1], [1,1,0,1,1,1], [1,1,1,0,0,0], [0,0,0,1,1,1],
[1,0,1,1,1,1], [1,1,1,1,0,1], [0,0,1,0,0,0], [0,0,0,1,0,0],
[1,0,0,1,1,0], [0,1,1,0,0,1], [1,1,0,0,0,0], [0,0,0,0,1,1],
[1,0,0,1,0,1], [1,0,1,0,0,1], [0,0,0,0,0,1], [1,0,0,0,0,0],
[1,0,0,1,1,1], [1,1,1,0,0,1], [1,0,0,0,0,1], [0,1,1,1,1,0],
[0,1,0,0,1,0], [1,0,1,1,0,1], [0,0,1,1,1,0], [0,1,1,1,0,0],
[0,0,1,1,1,1], [1,1,1,1,0,0], [0,0,0,1,0,1], [1,0,1,0,0,0],
[1,0,1,0,1,1], [1,1,0,1,0,1], [0,0,1,0,1,0], [0,1,0,1,0,0],
[1,1,0,0,0,1], [1,0,0,0,1,1], [1,1,1,1,1,0], [0,1,1,1,1,1],
[0,0,0,1,1,0], [0,1,1,0,0,0], [0,1,0,1,1,0], [0,1,1,0,1,0],
[1,0,1,1,1,0], [0,1,1,1,0,1], [1,0,0,1,0,0], [0,0,1,0,0,1],
[0,0,1,0,1,1], [1,1,0,1,0,0], [1,0,1,1,0,0], [0,0,1,1,0,1],
[0,1,1,0,1,1], [1,1,0,1,1,0], [0,1,0,0,1,1], [1,1,0,0,1,0],
[1,1,0,0,1,1], [0,0,1,1,0,0], [1,0,1,0,1,0], [0,1,0,1,0,1]
]
sigua_codes = [
[3, 3, 3], [0, 0, 0], [2, 0, 2], [1, 0, 1], [3, 2, 2], [1, 1, 3], [1, 0, 0], [0, 0, 2],
[3, 2, 3], [3, 1, 3], [3, 2, 0], [0, 1, 3], [2, 3, 3], [3, 3, 1], [0, 2, 0], [0, 1, 0],
[2, 1, 2], [1, 2, 1], [3, 0, 0], [0, 0, 3], [2, 1, 1], [2, 2, 1], [0, 0, 1], [2, 0, 0],
[2, 1, 3], [3, 2, 1], [2, 0, 1], [1, 3, 2], [1, 0, 2], [2, 3, 1], [0, 3, 2], [1, 3, 0],
[0, 3, 3], [3, 3, 0], [0, 1, 1], [2, 2, 0], [2, 2, 3], [3, 1, 1], [0, 2, 2], [1, 1, 0],
[3, 0, 1], [2, 0, 3], [3, 3, 2], [1, 3, 3], [0, 1, 2], [1, 2, 0], [1, 1, 2], [1, 2, 2],
[2, 3, 2], [1, 3, 1], [2, 1, 0], [0, 2, 1], [0, 2, 3], [3, 1, 0], [2, 3, 0], [0, 3, 1],
[1, 2, 3], [3, 1, 2], [1, 0, 3], [3, 0, 2], [3, 0, 3], [0, 3, 0], [2, 2, 2], [1, 1, 1]
]
bagua_codes = [
[0,0], [7,7],[3,5],[5,6],[0,5],[5,0],[5,7],[7,5],
[0,4], [1,0], [0,7], [7,0], [2,0], [0,2], [6,7], [7,3],
[3,1], [4,6], [1,7], [7,4], [3,2], [2,6], [7,6], [3,7],
[3,0],[0,6],[3,6],[4,1],[5,5],[2,2],[6,1],[4,3],
[6,0],[0,3],[7,2],[2,7],[2,4],[1,2],[6,5],[5,3],
[1,6],[3,4],[0,1],[4,0],[7,1],[4,7],[5,1],[4,5],
[2,1],[4,2],[3,3],[6,6],[6,4],[1,3],[2,3],[6,2],
[4,4],[1,1],[5,4],[1,5],[1,4],[6,3],[2,5],[5,2]
]
class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                    codebook_size: int = 1024,
                    token_size: int = 256,
                    commitment_cost: float = 0.25,
                    use_l2_norm: bool = False,
                    clustering_vq: bool = False,
                    block_size: int = 1
                    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_size = token_size
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.use_l2_norm = use_l2_norm

        self.clustering_vq = clustering_vq
        self.block_size = block_size
        if clustering_vq:
            self.decay = 0.99
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))

    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')
        z_flattened = z_flattened.reshape(-1, self.block_size, self.token_size)
        unnormed_z_flattened = z_flattened

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
        
        z_quantized_list = []
        encoding_indices_list = []
        
        for i in range(self.block_size):
            d = torch.sum(z_flattened[:,i,:]**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened[:,i,:], embedding.T)

            min_encoding_indice = torch.argmin(d, dim=1) # num_ele
            z_quantized = self.get_codebook_entry(min_encoding_indice)
            z_quantized_list.append(z_quantized)
            encoding_indices_list.append(min_encoding_indice)

        z_quantized = torch.stack(z_quantized_list, dim=1).view(z.shape)
        min_encoding_indices = torch.stack(encoding_indices_list, dim=1).view(-1, self.block_size)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding (handled by ResidualVectorQuantizer when used)
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        if self.clustering_vq and self.training:
            with torch.no_grad():
                # Gather distance matrix from all GPUs.
                
                if len(min_encoding_indices.shape) == 1:
                    encoding_indices = gather(min_encoding_indices)
                elif len(min_encoding_indices.shape) == 2:
                    encoding_indices = gather(min_encoding_indices.view(-1))
                else:
                    raise ValueError(f"min_encoding_indices in a wrong shape, {min_encoding_indices.shape}")
                # Compute and update the usage of each entry in the codebook.
                encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
                encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
                avg_probs = torch.mean(encodings, dim=0)
                self.embed_prob.mul_(self.decay).add_(avg_probs, alpha=1-self.decay)
                # Closest sampling to update the codebook.
                all_d = gather(d)
                all_unnormed_z_flattened = gather(unnormed_z_flattened).detach()
                if all_d.shape[0] != all_unnormed_z_flattened.shape[0]:
                    raise ValueError(
                        "all_d and all_unnormed_z_flattened have different length" + 
                        f"{all_d.shape}, {all_unnormed_z_flattened.shape}")
                indices = torch.argmin(all_d, dim=0)
                random_feat = all_unnormed_z_flattened[indices]
                # Decay parameter based on the average usage.
                decay = torch.exp(-(self.embed_prob * self.codebook_size * 10) /
                                    (1 - self.decay) - 1e-3).unsqueeze(1).repeat(1, self.token_size)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        emb = self.embedding.to(indices.device)
        if len(indices.shape) == 1:
            z_quantized = emb(indices)
        elif len(indices.shape) == 2:
            z_quantized = []
            for i in range(indices.shape[1]):
                z_quantized.append(emb(indices[:, i]))
            z_quantized = torch.cat(z_quantized, dim=1)
        else:
            raise NotImplementedError
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized


class ResidualVectorQuantizer(torch.nn.Module):
    """
    Residual vector quantizer composed of multiple VectorQuantizer layers.
    This implements four layers with configurable codebook sizes and block counts.
    For each layer i, the input feature dimension is split into `blocks[i]` parts
    (reshape to (num_ele, blocks, sub_dim)). Each part is quantized independently
    by a VectorQuantizer (with token_size=sub_dim). The layer's quantized parts
    are concatenated back to the original feature dimension. Residual
    quantization is performed sequentially: each layer quantizes the remaining
    residual and the reconstruction is accumulated.
    """
    def __init__(self,
                    codebook_sizes: List[int] = None,
                    blocks: List[int] = None,
                    token_size = 256,
                    commitment_cost: float = 0.25,
                    use_l2_norm: bool = False,
                    clustering_vq: bool = False):
        super().__init__()
        self.codebook_sizes = codebook_sizes if codebook_sizes is not None else [2, 4, 8, 64]
        self.blocks = blocks if blocks is not None else [6, 3, 2, 1]
        if len(self.codebook_sizes) != len(self.blocks):
            raise ValueError("codebook_sizes and blocks must have the same length")
        self.n_layers = len(self.codebook_sizes)
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm
        self.clustering_vq = clustering_vq

        # Layers will be lazily created on first forward pass because token_size
        # (sub-dimension per block) depends on the input channel dimension.
        self.vq_layers = torch.nn.ModuleList()

        for i in range(self.n_layers):
            block = self.blocks[i]
            if token_size % block != 0:
                raise ValueError(f"channels ({token_size}) must be divisible by block ({block}) for layer {i}")
            sub_dim = token_size // block
            vq = VectorQuantizer(codebook_size=self.codebook_sizes[i],
                                    token_size=sub_dim,
                                    commitment_cost=self.commitment_cost,
                                    use_l2_norm=self.use_l2_norm,
                                    clustering_vq=self.clustering_vq,
                                    block_size=block)
            self.vq_layers.append(vq)

    def clip_loss(self, logits, device):
        n = logits.shape[1]
        labels = torch.arange(n).to(device)
        # assert not torch.any(torch.isnan(logits))
        loss_i = F.cross_entropy(logits.transpose(0, 1), labels, reduction="mean")
        loss_t = F.cross_entropy(logits, labels, reduction="mean")
        loss = (loss_i + loss_t) / 2
        return loss

    def gua_loss(self, device):
        liusi = torch.arange(64).to(device)
        er_si = torch.tensor([
            [3,3,3],[0,0,0],[2,0,2],[1,0,1],[3,2,2],[1,1,3],[1,0,0],[0,0,2],
            [3,2,3],[3,1,3],[3,2,0],[0,1,3],[2,3,3],[3,3,1],[0,2,0],[0,1,0],
            [2,1,2],[1,2,1],[3,0,0],[0,0,3],[2,1,1],[2,2,1],[0,0,1],[2,0,0],
            [2,1,3],[3,2,1],[2,0,1],[1,3,2],[1,0,2],[2,3,1],[0,3,2],[1,3,0],
            [0,3,3],[3,3,0],[0,1,1],[2,2,0],[2,2,3],[3,1,1],[0,2,2],[1,1,0],
            [3,0,1],[2,0,3],[3,3,2],[1,3,3],[0,1,2],[1,2,0],[1,1,2],[1,2,2],
            [2,3,2],[1,3,1],[2,1,0],[0,2,1],[0,2,3],[3,1,0],[2,3,0],[0,3,1],
            [1,2,3],[3,1,2],[1,0,3],[3,0,2],[3,0,3],[0,3,0],[2,2,2],[1,1,1]
        ], dtype=torch.long, device=device)
        er_ba = torch.tensor([
            [0,0],[7,7],[3,5],[5,6],[0,5],[5,0],[5,7],[7,5],
            [0,4],[1,0],[0,7],[7,0],[2,0],[0,2],[6,7],[7,3],
            [3,1],[4,6],[1,7],[7,4],[3,2],[2,6],[7,6],[3,7],
            [3,0],[0,6],[3,6],[4,1],[5,5],[2,2],[6,1],[4,3],
            [6,0],[0,3],[7,2],[2,7],[2,4],[1,2],[6,5],[5,3],
            [1,6],[3,4],[0,1],[4,0],[7,1],[4,7],[5,1],[4,5],
            [2,1],[4,2],[3,3],[6,6],[6,4],[1,3],[2,3],[6,2],
            [4,4],[1,1],[5,4],[1,5],[1,4],[6,3],[2,5],[5,2]
        ], dtype=torch.long, device=device)
        er_liusi = torch.tensor([[1,1,1,1,1,1], [0,0,0,0,0,0], [1,0,0,0,1,0], [0,1,0,0,0,1],
    [1,1,1,0,1,0], [0,1,0,1,1,1], [0,1,0,0,0,0], [0,0,0,0,1,0],
    [1,1,1,0,1,1], [1,1,0,1,1,1], [1,1,1,0,0,0], [0,0,0,1,1,1],
    [1,0,1,1,1,1], [1,1,1,1,0,1], [0,0,1,0,0,0], [0,0,0,1,0,0],
    [1,0,0,1,1,0], [0,1,1,0,0,1], [1,1,0,0,0,0], [0,0,0,0,1,1],
    [1,0,0,1,0,1], [1,0,1,0,0,1], [0,0,0,0,0,1], [1,0,0,0,0,0],
    [1,0,0,1,1,1], [1,1,1,0,0,1], [1,0,0,0,0,1], [0,1,1,1,1,0],
    [0,1,0,0,1,0], [1,0,1,1,0,1], [0,0,1,1,1,0], [0,1,1,1,0,0],
    [0,0,1,1,1,1], [1,1,1,1,0,0], [0,0,0,1,0,1], [1,0,1,0,0,0],
    [1,0,1,0,1,1], [1,1,0,1,0,1], [0,0,1,0,1,0], [0,1,0,1,0,0],
    [1,1,0,0,0,1], [1,0,0,0,1,1], [1,1,1,1,1,0], [0,1,1,1,1,1],
    [0,0,0,1,1,0], [0,1,1,0,0,0], [0,1,0,1,1,0], [0,1,1,0,1,0],
    [1,0,1,1,1,0], [0,1,1,1,0,1], [1,0,0,1,0,0], [0,0,1,0,0,1],
    [0,0,1,0,1,1], [1,1,0,1,0,0], [1,0,1,1,0,0], [0,0,1,1,0,1],
    [0,1,1,0,1,1], [1,1,0,1,1,0], [0,1,0,0,1,1], [1,1,0,0,1,0],
    [1,1,0,0,1,1], [0,0,1,1,0,0], [1,0,1,0,1,0], [0,1,0,1,0,1]]).to(device)
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        liusigua = self.vq_layers[3].get_codebook_entry(liusi).reshape(64, -1)  # 64卦
        bagua = self.vq_layers[2].get_codebook_entry(er_ba).reshape(64, -1)  # 64卦
        sixiang = self.vq_layers[1].get_codebook_entry(er_si).reshape(64, -1)  # 64卦
        er_liusigua = self.vq_layers[0].get_codebook_entry(er_liusi).reshape(64, -1)  # 64卦

        logits_er_liusigua_liusigua = er_liusigua @ liusigua.t()
        logits_er_bagua_bagua = bagua @ liusigua.t()
        logits_er_sixiang_sixiang = sixiang @ liusigua.t()
        logits_sixiang_bagua_bagua = bagua @ sixiang.t()
        logits_er_liusigua_bagua = er_liusigua @ bagua.t()
        logits_er_liusigua_sixiang = er_liusigua @ sixiang.t()
    
        loss = loss + self.clip_loss(logits_er_liusigua_liusigua, device)\
            + self.clip_loss(logits_er_bagua_bagua, device)\
            + self.clip_loss(logits_er_sixiang_sixiang, device)\
            + self.clip_loss(logits_sixiang_bagua_bagua, device)\
            + self.clip_loss(logits_er_liusigua_bagua, device)\
            + self.clip_loss(logits_er_liusigua_sixiang, device)
        return loss

    def cuozong_loss(self, device):
        gua_codes = torch.tensor([
    [1,1,1,1,1,1], [0,0,0,0,0,0], [1,0,0,0,1,0], [0,1,0,0,0,1],
    [1,1,1,0,1,0], [0,1,0,1,1,1], [0,1,0,0,0,0], [0,0,0,0,1,0],
    [1,1,1,0,1,1], [1,1,0,1,1,1], [1,1,1,0,0,0], [0,0,0,1,1,1],
    [1,0,1,1,1,1], [1,1,1,1,0,1], [0,0,1,0,0,0], [0,0,0,1,0,0],
    [1,0,0,1,1,0], [0,1,1,0,0,1], [1,1,0,0,0,0], [0,0,0,0,1,1],
    [1,0,0,1,0,1], [1,0,1,0,0,1], [0,0,0,0,0,1], [1,0,0,0,0,0],
    [1,0,0,1,1,1], [1,1,1,0,0,1], [1,0,0,0,0,1], [0,1,1,1,1,0],
    [0,1,0,0,1,0], [1,0,1,1,0,1], [0,0,1,1,1,0], [0,1,1,1,0,0],
    [0,0,1,1,1,1], [1,1,1,1,0,0], [0,0,0,1,0,1], [1,0,1,0,0,0],
    [1,0,1,0,1,1], [1,1,0,1,0,1], [0,0,1,0,1,0], [0,1,0,1,0,0],
    [1,1,0,0,0,1], [1,0,0,0,1,1], [1,1,1,1,1,0], [0,1,1,1,1,1],
    [0,0,0,1,1,0], [0,1,1,0,0,0], [0,1,0,1,1,0], [0,1,1,0,1,0],
    [1,0,1,1,1,0], [0,1,1,1,0,1], [1,0,0,1,0,0], [0,0,1,0,0,1],
    [0,0,1,0,1,1], [1,1,0,1,0,0], [1,0,1,1,0,0], [0,0,1,1,0,1],
    [0,1,1,0,1,1], [1,1,0,1,1,0], [0,1,0,0,1,1], [1,1,0,0,1,0],
    [1,1,0,0,1,1], [0,0,1,1,0,0], [1,0,1,0,1,0], [0,1,0,1,0,1]
]).to(device)
        er_gua = self.vq_layers[0].get_codebook_entry(gua_codes).reshape(64, -1)
        er_cuogua = self.vq_layers[0].get_codebook_entry(1 - gua_codes).reshape(64, -1)
        er_zonggua = self.vq_layers[0].get_codebook_entry(gua_codes.flip(-1)).reshape(64, -1)
        er_cuozonggua = self.vq_layers[0].get_codebook_entry(1 - gua_codes.flip(-1)).reshape(64, -1)
        er_cuoloss = self.clip_loss(er_gua @ er_cuogua.t(), device)
        er_zongloss = self.clip_loss(er_gua @ er_zonggua.flip(-1).t(), device)
        er_cuozongloss = self.clip_loss(er_gua @ er_cuozonggua.t(), device)
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)
        loss = loss - er_cuoloss + er_zongloss + er_cuozongloss
        return loss

    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        b, c, h, w = z.shape
        z_flattened = z

        residual_flat = z_flattened
        recon_flat = torch.zeros_like(z_flattened)
        
        overall_min_list = []
        total_commitment_loss = torch.tensor(0., device=z.device)
        total_codebook_loss = torch.tensor(0., device=z.device)

        # Process each residual VQ layer sequentially
        for i, vq in enumerate(self.vq_layers):

            q_slice, rdict = vq(residual_flat)
                # NOTE: do not sum per-slice losses here. We'll compute per-layer
                # commitment/codebook losses after concatenating the quantized slices.

                # collect min indices
            min_idx = rdict.get('min_encoding_indices')
            total_commitment_loss = total_commitment_loss + rdict['commitment_loss']
            total_codebook_loss = total_codebook_loss + rdict['codebook_loss']
            
            quant_layer_flat = q_slice
            # accumulate reconstruction and update residual
            recon_flat = recon_flat + quant_layer_flat
            residual_flat = residual_flat - quant_layer_flat

                # map per-slice digits -> index in the corresponding code table
            if i == 3:
                overall_min_list.append(min_idx.squeeze(1))
            else:
                if i == 0:
                    table = torch.tensor(gua_codes, device=min_idx.device)  # (64, 6)
                elif i == 1:
                    table = torch.tensor(sigua_codes, device=min_idx.device)  # (64, 3)
                elif i == 2:
                    table = torch.tensor(bagua_codes, device=min_idx.device)  # (64, 2)
                # digits shape must match table width
                if min_idx.shape[1] != table.shape[1]:
                    raise ValueError(f"Layer {i}: expected {table.shape[1]} slices, got {min_idx.shape[1]}")

                digits_flat = min_idx.reshape(-1, min_idx.shape[1])

                # compare each row in digits_flat to all rows in table:
                # digits_flat: (num_ele, slices), table: (64, slices)
                # -> matches: (num_ele, 64, slices)
                matches = (digits_flat.unsqueeze(1) == table.unsqueeze(0))
                mask = matches.all(dim=2).int()  # (num_ele, 64)

                if not torch.any(mask):
                    raise ValueError(f"No matching codebook entry found for layer {i}")
                # ensure unique match per element
                if torch.any(mask.sum(dim=1) != 1):
                    raise ValueError(f"Non-unique or missing matches found for layer {i}")
                indices_flat = torch.argmax(mask, dim=1)  # (num_ele,)

                indices = indices_flat.view(min_idx.size(0))
                overall_min_list.append(indices)

        # reshape reconstruction back to original shape
        z_quantized = recon_flat

        loss = total_commitment_loss + total_codebook_loss
        
        if self.training:
            loss = loss + self.gua_loss(z.device) + self.cuozong_loss(z.device)

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=total_commitment_loss,
            codebook_loss=total_codebook_loss,
                # stack min indices across layers: shape (b, n_layers, h, w)
            min_encoding_indices=torch.stack(overall_min_list, dim=1).view(b, self.n_layers, h, w)
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        """Reconstruct quantized tensor from per-layer indices.

        Accepts `indices` with shape either `(b, n_layers, h, w)` or
        `(num_ele, n_layers)`. For the first three layers the provided index
        refers to an entry in the corresponding code table (`gua_codes`,
        `sigua_codes`, `bagua_codes`) which yields per-slice code indices; for
        later layers indices are treated as direct codebook indices.

        Returns a tensor shaped `(b, c, h, w)` when input is 4D, otherwise
        returns `(num_ele, c)`.
        """
        arr = indices
        # normalize to (num_ele, n_layers)
        if arr.dim() == 4:
            b, n_layers, h, w = arr.shape
            flat = arr.permute(0, 2, 3, 1).reshape(-1, n_layers)  # (num_ele, n_layers)
            return_4d = True
        elif arr.dim() == 2:
            flat = arr
            return_4d = False
            # num_ele unknown here
            b = None
            h = None
            w = None
            n_layers = flat.shape[1]
        else:
            raise ValueError("indices must be shape (b, n_layers, h, w) or (num_ele, n_layers)")

        num_ele = flat.shape[0]
        n_layers = flat.shape[1]
        if n_layers != self.n_layers:
            raise ValueError(f"expected {self.n_layers} layers, got {n_layers}")

        device = flat.device
        summed = None

        for i in range(self.n_layers):
            vq = self.vq_layers[i]

            idx_layer = flat[:, i] #.long()  # (num_ele,)

            if i == 0:
                table = torch.tensor(gua_codes, device=device)
                digits = table[idx_layer.view(-1)]  # (num_ele, block)
            elif i == 1:
                table = torch.tensor(sigua_codes, device=device)
                digits = table[idx_layer.view(-1)]
            elif i == 2:
                table = torch.tensor(bagua_codes, device=device)
                digits = table[idx_layer.view(-1)]
            else:
                # layer with no table: treat idx_layer as direct per-slice indices
                digits = idx_layer #.view(-1, 1)

            layer_feat = self.vq_layers[i].get_codebook_entry(digits)

            if summed is None:
                summed = layer_feat
            else:
                summed = summed + layer_feat

        if return_4d:
            zq = summed.view(b, h, w).contiguous()
            return zq
        else:
            return summed


class DiagonalGaussianDistribution(object):
    @autocast(enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @autocast(enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @autocast(enabled=False)
    def mode(self):
        return self.mean

    @autocast(enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean.float(), 2)
                                    + self.var.float() - 1.0 - self.logvar.float(),
                                    dim=[1, 2])
