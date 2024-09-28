import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange

try:
    from .attention_utils import get_embeddings
except:
    from attention_utils import get_embeddings
from logging import getLogger

logger = getLogger()


def get_embedder(config, x_num, max_output_dim):
    if config.type == "linear":
        embedder = LinearEmbedder
    elif config.type == "conv":
        embedder = ConvEmbedder
    else:
        raise ValueError(f"Unknown embedder type: {config.type}")

    return embedder(config, x_num, max_output_dim)


def patchify(data: torch.Tensor, patch_num: int):
    """
    Input:
        (bs, nt, px, py, d)
    Output:
        (bs, nt, p*p, x*y*d)
    """
    bs, nt, px, py, d = data.size()
    p = patch_num
    x = px // p
    y = py // p

    data = data.view(bs, nt, p, x, p, y, d).permute(
        0, 1, 2, 4, 3, 5, 6
    )  # (bs, nt, p, x, p, y, d) -> (bs, nt, p, p, x, y, d)

    data = data.reshape((bs, nt, p * p, x * y * d))
    return data


def depatchify(data: torch.Tensor, patch_num: int, x: int, y: int, d: int):
    """
    Input:
        (bs, nt, p*p, x*y*d)
    Output:
        (bs, nt, px, py, d)
    """
    bs = data.size(0)
    nt = data.size(1)
    p = patch_num

    data = data.view(bs, nt, p, p, x, y, d).permute(
        0, 1, 2, 4, 3, 5, 6
    )  # (bs, nt, p, p, x, y, d) -> (bs, nt, p, x, p, y, d)

    data = data.reshape((bs, nt, p * x, p * y, d))
    return data


def layer_initialize(layer, mode="zero", gamma=0.01):
    # re-initialize given layer to have small outputs
    if mode == "zero":
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif mode == "uniform":
        nn.init.uniform_(layer.weight, -gamma, gamma)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -gamma, gamma)
    else:
        raise ValueError(f"Unknown mode {mode}")


class LinearEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        # for encoder part
        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num * config.patch_num, self.dim))

        self.time_proj = nn.Sequential(
            nn.Linear(1, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.pre_proj = nn.Sequential(
            nn.Linear(self.patch_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

        # for decoder part

        self.post_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.patch_dim_output),
        )

    def encode(self, data, times):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs/1, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """
        bs = data.size(0)
        data = patchify(data, self.config.patch_num)  # (bs, input_len, p*p, x*y*d)
        data = self.pre_proj(data)  # (bs, input_len, p*p, dim)

        time_embeddings = self.time_proj(times)[:, :, None]  # (bs/1, input_len, 1, dim)
        data = ((data + time_embeddings) + self.patch_position_embeddings).reshape(bs, -1, self.dim)
        return data

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)

        data_output = self.post_proj(data_output)  # (bs, query_len, patch_dim)
        data_output = data_output.view(
            bs, -1, self.config.patch_num_output * self.config.patch_num_output, self.patch_dim_output
        )  # (bs, output_len, p*p, patch_dim)

        data_output = depatchify(
            data_output,
            self.config.patch_num_output,
            self.patch_resolution_output,
            self.patch_resolution_output,
            self.data_dim,
        )  # (bs, output_len, x_num, x_num, data_dim)

        return data_output


class ConvEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """

    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config.dim
        self.data_dim = data_dim

        assert (
            x_num % config.patch_num == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config.patch_num}"
        self.patch_resolution = x_num // config.patch_num  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config.patch_num_output == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config.patch_num_output}"
        self.patch_resolution_output = (
            x_num // config.patch_num_output
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        ## for encoder part

        self.patch_position_embeddings = get_embeddings((1, 1, config.patch_num * config.patch_num, self.dim))

        self.time_embed_type = config.get("time_embed", "continuous")
        if self.time_embed_type == "continuous":
            self.time_proj = nn.Sequential(
                nn.Linear(1, self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim),
            )
        else:
            self.time_embed = get_embeddings((1, config.get("max_time_len", 10), 1, self.dim))

        if config.get("early_conv", 0):
            n_conv_layers = math.log2(self.patch_resolution)
            assert n_conv_layers.is_integer(), f"patch_resolution {self.patch_resolution} must be a power of 2"
            n_conv_layers = int(n_conv_layers)
            kernel_size = [3] * n_conv_layers + [1]
            stride = [2] * n_conv_layers + [1]
            padding = [1] * n_conv_layers + [0]
            channels = [data_dim] + [self.dim // (2**i) for i in range(n_conv_layers - 1, 0, -1)] + [self.dim, self.dim]

            self.conv_proj = nn.Sequential()
            for i in range(len(kernel_size)):
                self.conv_proj.append(
                    nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size[i],
                        stride=stride[i],
                        padding=padding[i],
                    )
                )
                if i < len(kernel_size) - 1:
                    self.conv_proj.append(nn.GELU())
        else:
            # regular vit patch embedding
            self.conv_proj = nn.Sequential(
                nn.Conv2d(
                    in_channels=data_dim,
                    out_channels=self.dim,
                    kernel_size=self.patch_resolution,
                    stride=self.patch_resolution,
                ),
                nn.GELU(),
                nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1),
            )

        ## for decoder part

        self.conv_dim = config.get("conv_dim", self.dim // 4)

        if config.get("deep", 0):
            self.post_proj = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                nn.GELU(),
                nn.Linear(in_features=self.dim, out_features=self.dim),
                nn.GELU(),
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1),
            )
        else:
            self.post_proj = nn.Sequential(
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config.patch_num_output, w=self.config.patch_num_output),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                nn.GELU(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1),
            )

        if config.get("initialize_small_output", 0):
            layer_initialize(self.post_proj[-1], mode=config.initialize_small_output)

    def encode(self, data, times):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """

        bs = data.size(0)
        data = rearrange(data, "b t h w c -> (b t) c h w")
        data = self.conv_proj(data)  # (bs*input_len, d, patch_num, patch_num)
        data = rearrange(data, "(b t) d h w -> b t (h w) d", b=bs)  # (bs, input_len, p*p, dim)

        if self.time_embed_type == "continuous":
            time_embeddings = self.time_proj(times)[:, :, None]  # (bs, input_len, 1, dim)
        else:
            time_embeddings = self.time_embed[:, : times.size(1)]  # (bs, input_len, 1, dim)

        data = ((data + time_embeddings) + self.patch_position_embeddings).reshape(bs, -1, self.dim)
        return data

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)
        data_output = self.post_proj(data_output)  # (bs*output_len, data_dim, x_num, x_num)
        data_output = rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
        return data_output


if __name__ == "__main__":
    from omegaconf import OmegaConf

    conf = OmegaConf.load("../configs/model/prose_2to1.yaml")
    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    bs = 4
    input_len = 10
    x_num = 128
    data_dim = 4

    embedder = ConvEmbedder(conf.embedder, x_num, data_dim)
    embedder.to(device)

    data = torch.randn(bs, input_len, x_num, x_num, data_dim, device=device)
    times = torch.randn(bs, input_len, 1, device=device)

    # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
    data_input = embedder.encode(data, times)
    print(f"{data_input.size() = }")

    dim = conf.dim_emb
    # query_len = conf.embedder.patch_num_output * conf.embedder.patch_num_output
    query_len = conf.embedder.patch_num_output * (conf.embedder.patch_num_output // 2 + 1)

    data = torch.randn(bs, query_len, dim, device=device)
    data_output = embedder.decode(data)
    print(f"{data_output.size() = }")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{count_parameters(embedder):,}")
