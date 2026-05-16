import math
import torch
import importlib
import numpy as np
from torch.nn.functional import silu
from torch.utils.checkpoint import checkpoint
from typing import Any, List, Literal, Tuple, Union, Set, Dict

# 自定义模块
from torch_utils import persistence
from models import weight_init
from models import Linear, Conv2d, GroupNorm, AttentionOp
from models import UNetBlock
from models import PositionalEmbedding, FourierEmbedding


# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch


network_module = importlib.import_module("models")


@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=128,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 2, 2],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,  # Number of residual blocks per resolution.
        attn_resolutions=[16],  # List of resolutions with self-attention.
        dropout=0.10,  # Dropout probability of intermediate activations.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
        embedding_type="positional",  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type="standard",  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type="standard",  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter=[1, 1],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        assert embedding_type in ["fourier", "positional"]
        assert encoder_type in ["standard", "skip", "residual"]
        assert decoder_type in ["standard", "skip"]

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        self.img_resolution = img_resolution
        if isinstance(img_resolution, int):
            self.img_shape_y = self.img_shape_x = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        # Mapping.
        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = (
            Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        )
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            # res = img_resolution >> level
            res = self.img_shape_y >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [block.out_channels for name, block in self.enc.items() if "aux" not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            # res = img_resolution >> level
            res = self.img_shape_y >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero
                )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


# ------------------------------------------------------------------------------
# PhysicsnemoSongUNet


@persistence.persistent_class
class PhysicsnemoSongUNet(torch.nn.Module):
    r"""
    This architecture is a diffusion backbone for 2D image generation.
    It is a reimplementation of the `DDPM++
    <https://proceedings.mlr.press/v139/nichol21a.html>`_ and `NCSN++ <https://arxiv.org/abs/2011.13456>`_
    architectures, which are U-Net variants
    with optional self-attention, embeddings, and encoder-decoder components.

    This model supports conditional and unconditional setups, as well as several
    options for various internal architectural choices such as encoder and decoder
    type, embedding type, etc., making it flexible and adaptable to different tasks
    and configurations.

    This architecture supports conditioning on the noise level (called *noise labels*),
    as well as on additional vector-valued labels (called *class labels*) and (optional)
    vector-valued augmentation labels. The conditioning mechanism relies on addition
    of the conditioning embeddings in the U-Net blocks of the encoder. To condition
    on images, the simplest mechanism is to concatenate the image to the input
    before passing it to the SongUNet.

    The model first applies a mapping operation to generate embeddings for all
    the conditioning inputs (the noise level, the class labels, and the
    optional augmentation labels).

    Then, at each level in the U-Net encoder, a sequence of blocks is applied:

    • A first block downsamples the feature map resolution by a factor of 2
      (odd resolutions are floored). This block does not change the number of
      channels.

    • A sequence of ``num_blocks`` U-Net blocks are applied, each with a different
      number of channels. These blocks do not change the feature map
      resolution, but they multiply the number of channels by a factor
      specified in ``channel_mult``.
      If required, the U-Net blocks also apply self-attention at the specified
      resolutions.

    • At the end of the level, the feature map is cached to be used in a skip
      connection in the decoder.

    The decoder is a mirror of the encoder, with the same number of levels and
    the same number of blocks per level. It multiplies the feature map resolution
    by a factor of 2 at each level.

    Parameters
    -----------
    img_resolution : Union[List[int, int], int]
        The resolution of the input/output image. Can be a single int :math:`H` for
        square images or a list :math:`[H, W]` for rectangular images.

        *Note:* This parameter is only used as a convenience to build the
        network. In practice, the model can still be used with images of
        different resolutions. The only exception to this rule is when
        ``additive_pos_embed`` is True, in which case the resolution of the latent
        state :math:`\mathbf{x}` must match ``img_resolution``.
    in_channels : int
        Number of channels :math:`C_{in}` in the input image. May include channels from both
        the latent state and additional channels when conditioning on images.
        For an unconditional model, this should be equal to ``out_channels``.
    out_channels : int
        Number of channels :math:`C_{out}` in the output image. Should be equal to the number
        of channels :math:`C_{\mathbf{x}}` in the latent state.
    label_dim : int, optional, default=0
        Dimension of the vector-valued ``class_labels`` conditioning; 0
        indicates no conditioning on class labels.
    augment_dim : int, optional, default=0
        Dimension of the vector-valued `augment_labels` conditioning; 0 means
        no conditioning on augmentation labels.
    model_channels : int, optional, default=128
        Base multiplier for the number of channels accross the entire network.
    channel_mult : List[int], optional, default=[1, 2, 2, 2]
        Multipliers for the number of channels at every level in
        the encoder and decoder. The length of ``channel_mult`` determines the
        number of levels in the U-Net. At level ``i``, the number of channel in
        the feature map is ``channel_mult[i] * model_channels``.
    channel_mult_emb : int, optional, default=4
        Multiplier for the number of channels in the embedding vector. The
        embedding vector has ``model_channels * channel_mult_emb`` channels.
    num_blocks : int, optional, default=4
        Number of U-Net blocks at each level.
    attn_resolutions : List[int], optional, default=[16]
        Resolutions of the levels at which self-attention layers are applied.
        Note that the feature map resolution must match exactly the value
        provided in `attn_resolutions` for the self-attention layers to be
        applied.
    dropout : float, optional, default=0.10
        Dropout probability applied to intermediate activations within the
        U-Net blocks.
    label_dropout : float, optional, default=0.0
        Dropout probability applied to the `class_labels`. Typically used for
        classifier-free guidance.
    embedding_type : Literal["fourier", "positional", "zero"], optional, default="positional"
        Diffusion timestep embedding type: 'positional' for DDPM++, 'fourier'
        for NCSN++, 'zero' for none.
    channel_mult_noise : int, optional, default=1
        Multiplier for the number of channels in the noise level embedding. The
        noise level embedding vector has ``model_channels * channel_mult_noise`` channels.
    encoder_type : Literal["standard", "skip", "residual"], optional, default="standard"
        Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++, 'skip' for skip connections.
    decoder_type : Literal["standard", "skip"], optional, default="standard"
        Decoder architecture: 'standard' or 'skip' for skip connections.
    resample_filter : List[int], optional, default=[1, 1]
        Resampling filter coefficients applied in the U-Net blocks
        convolutions: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    checkpoint_level : int, optional, default=0
        Number of levels that should use gradient checkpointing. Only levels at
        which the feature map resolution is large enough will be checkpointed
        (0 disables checkpointing, higher values means more layers are checkpointed).
        Higher values trade memory for computation.
    additive_pos_embed : bool, optional, default=False
        If ``True``, adds a learnable positional embedding after the first convolution layer.
        Used in StormCast model.

        *Note:* Those positional embeddings encode spatial position information
        of the image pixels, unlike the ``embedding_type`` parameter which encodes
        temporal information about the diffusion process. In that sense it is a
        simpler version of the positional embedding used in
        :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`.

        如果为“True”，则在第一个卷积层之后添加一个可学习的位置嵌入。用于StormCast模型。
        *注意：*这些位置嵌入对图像像素的空间位置信息进行编码，这与对扩散过程的时间信息进行编码的“embedding_type”参数不同。
        从这个意义上说，它是：class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`中使用的位置嵌入的一个更简单的版本。
    use_apex_gn : bool, optional, default=False
        A flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Apex needs to be installed for this to work. Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
        Required when ``use_apex_gn`` is ``True``.
    profile_mode : bool, optional, default=False
        A flag indicating whether to enable all nvtx annotations during
        profiling.
    amp_mode : bool, optional, default=False
        A flag indicating whether mixed-precision (AMP) training is enabled.


    Forward
    -------
    x : torch.Tensor
        The input image of shape :math:`(B, C_{in}, H_{in}, W_{in})`. In
        general ``x`` is the channel-wise concatenation of the latent state
        :math:`\mathbf{x}` and additional images used for conditioning. For an
        unconditional model, ``x`` is simply the latent state
        :math:`\mathbf{x}`.

        *Note:* :math:`H_{in}` and :math:`W_{in}` do not need to match
        :math:`H` and :math:`W` defined in ``img_resolution``, except when
        ``additive_pos_embed`` is ``True``. In that case, the resolution of
        ``x`` must match ``img_resolution``.
    noise_labels : torch.Tensor
        The noise labels of shape :math:`(B,)`. Used for conditioning on
        the diffusion noise level.
    class_labels : torch.Tensor
        The class labels of shape :math:`(B, \text{label_dim})`. Used for
        conditioning on any vector-valued quantity. Can pass ``None`` when
        ``label_dim`` is 0.
    augment_labels : torch.Tensor, optional, default=None
        The augmentation labels of shape :math:`(B, \text{augment_dim})`. Used
        for conditioning on any additional vector-valued quantity. Can pass
        ``None`` when ``augment_dim`` is 0.

    Outputs
    -------
    torch.Tensor
        The denoised latent state of shape :math:`(B, C_{out}, H_{in}, W_{in})`.


    .. important::
        • The terms *noise levels* (or *noise labels*) are used to refer to the diffusion time-step, as these are conceptually equivalent.
        • The terms *labels* and *classes* originate from the original paper and EDM repository,
          where this architecture was used for class-conditional image generation. While these terms
          suggest class-based conditioning, the architecture can actually be conditioned on any vector-valued
          conditioning.
        • The term *positional embedding* used in the `embedding_type` parameter
          also comes from the original paper and EDM repository. Here,
          *positional* refers to the diffusion time-step, similar to how position is used in transformer
          architectures. Despite the name, these embeddings encode temporal information about the
          diffusion process rather than spatial position information.
        • Limitations on input image resolution: for a model that has :math:`N` levels,
          the latent state :math:`\mathbf{x}` must have resolution that is a multiple of :math:`2^N` in each dimension.
          This is due to a limitation in the decoder that does not support shape mismatch
          in the residual connections from the encoder to the decoder. For images that do not match
          this requirement, it is recommended to interpolate your data on a grid of the required resolution
          beforehand.

    Example
    --------
    >>> model = SongUNet(img_resolution=16, in_channels=2, out_channels=2)
    >>> noise_labels = torch.randn([1])
    >>> class_labels = torch.randint(0, 1, (1, 1))
    >>> input_image = torch.ones([1, 2, 16, 16])
    >>> output_image = model(input_image, noise_labels, class_labels)
    >>> output_image.shape
    torch.Size([1, 2, 16, 16])
    """

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method.

    def __init__(
        self,
        img_resolution: Union[List[int], int],
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2],
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: List[int] = [16],
        dropout: float = 0.10,
        label_dropout: float = 0.0,
        embedding_type: Literal["fourier", "positional", "zero"] = "positional",
        channel_mult_noise: int = 1,
        encoder_type: Literal["standard", "skip", "residual"] = "standard",
        decoder_type: Literal["standard", "skip"] = "standard",
        resample_filter: List[int] = [1, 1],
        checkpoint_level: int = 0,
        additive_pos_embed: bool = False,
        use_apex_gn: bool = False,
        act: str = "silu",
        profile_mode: bool = False,
        amp_mode: bool = False,
    ):
        valid_embedding_types = ["fourier", "positional", "zero"]
        if embedding_type not in valid_embedding_types:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Must be one of {valid_embedding_types}.")

        valid_encoder_types = ["standard", "skip", "residual"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}.")

        valid_decoder_types = ["standard", "skip"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}.")

        super().__init__()
        self.label_dropout = label_dropout
        self.embedding_type = embedding_type
        emb_channels = model_channels * channel_mult_emb
        self.emb_channels = emb_channels
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=0.7071067811865476,  # 1 / sqrt(2)
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        # for compatibility with older versions that took only 1 dimension
        self.img_resolution = img_resolution
        if isinstance(img_resolution, int):
            self.img_shape_y = self.img_shape_x = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        self._num_levels = len(channel_mult)
        self._input_shape_mult = 2**self._num_levels

        # set the threshold for checkpointing based on image resolution
        self.checkpoint_threshold = (math.floor(math.sqrt(self.img_shape_x * self.img_shape_y)) >> checkpoint_level) + 1

        # Optional additive learned positition embed after the first conv
        self.additive_pos_embed = additive_pos_embed
        if self.additive_pos_embed:
            self.spatial_emb = torch.nn.Parameter(torch.randn(1, model_channels, self.img_shape_y, self.img_shape_x))
            torch.nn.init.trunc_normal_(self.spatial_emb, std=0.02)

        # Mapping.
        if self.embedding_type != "zero":
            self.map_noise = (
                PositionalEmbedding(num_channels=noise_channels, endpoint=True)
                if embedding_type == "positional"
                else FourierEmbedding(num_channels=noise_channels)
            )
            self.map_label = (
                Linear(
                    in_features=label_dim,
                    out_features=noise_channels,
                    **init,
                )
                if label_dim
                else None
            )
            self.map_augment = (
                Linear(
                    in_features=augment_dim,
                    out_features=noise_channels,
                    bias=False,
                    **init,
                )
                if augment_dim
                else None
            )
            self.map_layer0 = Linear(
                in_features=noise_channels,
                out_features=emb_channels,
                **init,
            )
            self.map_layer1 = Linear(
                in_features=emb_channels,
                out_features=emb_channels,
                **init,
            )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_shape_y >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=cin,
                    out_channels=cout,
                    kernel=3,
                    **init,
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}x{res}_aux_down"] = Conv2d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}x{res}_aux_skip"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=1,
                        **init,
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}x{res}_aux_residual"] = Conv2d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
        skips = [block.out_channels for name, block in self.enc.items() if "aux" not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.img_shape_y >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                )
            if decoder_type == "skip" or level == 0:
                if decoder_type == "skip" and level < len(channel_mult) - 1:
                    self.dec[f"{res}x{res}_aux_up"] = Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel=0,
                        up=True,
                        resample_filter=resample_filter,
                    )
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(
                    in_channels=cout,
                    out_channels=out_channels,
                    kernel=3,
                    **init_zero,
                )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Validate input shapes
        batch_size = x.shape[0]

        if x.ndim != 4:
            raise ValueError(f"Expected 'x' to be a 4D tensor, " f"got {x.ndim}D tensor with shape {tuple(x.shape)}")

        # Check spatial dimensions are powers of 2 or multiples of 2^N
        for d in x.shape[-2:]:
            # Check if d is a power of 2
            is_power_of_2 = (d & (d - 1)) == 0 and d > 0
            # If not power of 2, must be multiple of self._input_shape_mult
            if not ((is_power_of_2 and d < self._input_shape_mult) or (d % self._input_shape_mult == 0)):
                raise ValueError(
                    f"Input spatial dimensions ({x.shape[-2:]}) must be "
                    f"either powers of 2 or multiples of 2**N where "
                    f"N (={self._num_levels}) is the number of levels "
                    f"in the U-Net."
                )

        # TODO: noise_labels of shape (1,) means that all inputs share the
        # same noise level. This should be removed in the future, though.
        if noise_labels.ndim != 1 or noise_labels.shape[0] not in (batch_size, 1):
            raise ValueError(
                f"Expected 'noise_labels' shape ({batch_size},) or (1,), " f"got {tuple(noise_labels.shape)}"
            )

        if class_labels is not None and (class_labels.ndim != 2 or class_labels.shape[0] != batch_size):
            raise ValueError(f"Expected 'class_labels' shape ({batch_size}, C), " f"got {tuple(class_labels.shape)}")

        if augment_labels is not None and (augment_labels.ndim != 2 or augment_labels.shape[0] != batch_size):
            raise ValueError(
                f"Expected 'augment_labels' shape ({batch_size}, C), " f"got {tuple(augment_labels.shape)}"
            )

        if self.embedding_type != "zero":
            # Mapping.
            emb = self.map_noise(noise_labels)
            emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = silu(self.map_layer0(emb))
            emb = silu(self.map_layer1(emb))
        else:
            emb = torch.zeros(
                (noise_labels.shape[0], self.emb_channels),
                device=x.device,
                dtype=x.dtype,
            )

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            elif "_conv" in name:
                x = block(x)
                if self.additive_pos_embed:
                    x = x + self.spatial_emb.to(dtype=x.dtype)
                skips.append(x)
            else:
                # For UNetBlocks check if we should use gradient checkpointing
                if isinstance(block, UNetBlock):
                    if math.floor(math.sqrt(x.shape[-2] * x.shape[-1])) > self.checkpoint_threshold:
                        # self.checkpoint = checkpoint?
                        # else: self.checkpoint  = lambda(block,x,emb:block(x,emb))
                        x = checkpoint(block, x, emb, use_reentrant=False)
                    else:
                        # AssertionError: Only support NHWC layout.
                        x = block(x, emb)
                else:
                    x = block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_up" in name:
                aux = block(aux)
            elif "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                # check for checkpointing on decoder blocks and up sampling blocks
                if (
                    math.floor(math.sqrt(x.shape[-2] * x.shape[-1])) > self.checkpoint_threshold and "_block" in name
                ) or (
                    math.floor(math.sqrt(x.shape[-2] * x.shape[-1])) > (self.checkpoint_threshold / 2) and "_up" in name
                ):
                    x = checkpoint(block, x, emb, use_reentrant=False)
                else:
                    x = block(x, emb)
        return aux


# ------------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion


@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,  # Image resolution at input/output.
        in_channels,  # Number of color channels at input.
        out_channels,  # Number of color channels at output.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4],  # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        dropout=0.10,  # List of resolutions with self-attention.
        label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)
        block_kwargs = dict(
            emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = (
            Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero)
            if augment_dim
            else None
        )
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = (
            Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
            if label_dim
            else None
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs
                )
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x


@persistence.persistent_class
class CorrDiffRegressionUNet(torch.nn.Module):  # TODO a lot of redundancy, need to clean up
    r"""
    This interface provides a U-Net wrapper for CorrDiff deterministic
    regression model (and other deterministic downsampling models).
    It supports the following architectures:
    此接口为 CorrDiff 确定性回归模型（及其他确定性下采样模型）提供了一个 U-Net 封装。
    它支持以下架构：
    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNet`
    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`
    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`
    - :class:`~physicsnemo.models.diffusion.dhariwal_unet.DhariwalUNet`

    It shares the same architeture as a conditional diffusion model. It does so
    by concatenating a conditioning image to a zero-filled latent state, and by
    setting the noise level and the class labels to zero.
    它采用与条件扩散模型相同的架构。
    其实现方式是将条件图像与一个零填充的 latent state 进行拼接，并将噪声级别和类别标签设为零。

    Parameters
    -----------
    img_resolution : Union[int, Tuple[int, int]]
        The resolution of the input/output image. If a single int is provided,
        then the image is assumed to be square.
    img_in_channels : int
        Number of channels in the input image.
    img_out_channels : int
        Number of channels in the output image.
    use_fp16: bool, optional, default=False
        Execute the underlying model at FP16 precision.
    model_type: Literal['SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'], default='SongUNetPosEmbd'
        Class name of the underlying architecture. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
    **model_kwargs : dict
        Keyword arguments passed to the underlying architecture `__init__` method.

    Please refer to the documentation of these classes for details on how to call
    and use these models directly.

    Forward
    -------
    x : torch.Tensor
        The input tensor, typically zero-filled, of shape :math:`(B, C_{in}, H_{in}, W_{in})`.
    img_lr : torch.Tensor
        Conditioning image of shape :math:`(B, C_{lr}, H_{in}, W_{in})`.
    **model_kwargs : dict
        Additional keyword arguments to pass to the underlying architecture
        forward method.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H_{in}, W_{in})` (same
        spatial dimensions as the input).

    """

    # Classes that can be wrapped by this UNet class.
    _wrapped_classes: Set[str] = {"SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"}
    _wrapped_classes: Set[str] = {"SongUNet", "DhariwalUNet", "PhysicsnemoSongUNet"}

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal["SongUNet", "DhariwalUNet", "PhysicsnemoSongUNet"] = "PhysicsnemoSongUNet",
        **model_kwargs: dict,
    ):
        super().__init__()

        # Validation
        if model_type not in self._wrapped_classes:
            raise ValueError(
                f"Model type '{model_type}' is not supported. " f"Must be one of: {', '.join(self._wrapped_classes)}"
            )

        # for compatibility with older versions that took only 1 dimension
        if isinstance(img_resolution, int):
            self.img_shape_x = self.img_shape_y = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )
        self.use_fp16 = use_fp16

    @property
    def use_fp16(self):
        """
        bool: Whether the model uses float16 precision.

        Returns
        -------
        bool
            True if the model is in float16 mode, False otherwise.
        """
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        """
        Set whether the model should use float16 precision.

        Parameters
        ----------
        value : bool
            If True, moves the model to torch.float16. If False, moves to torch.float32.

        Raises
        ------
        ValueError
            If `value` is not a boolean.
        """
        # NOTE: allow 0/1 values for older checkpoints
        if not (isinstance(value, bool) or value in [0, 1]):
            raise ValueError(f"`use_fp16` must be a boolean, but got {type(value).__name__}.")
        self._use_fp16 = value
        if value:
            self.to(torch.float16)
        else:
            self.to(torch.float32)

    def forward(
        self,
        x: torch.Tensor,  # zero_input
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        # SR: concatenate input channels
        # 对应 EDMPrecondSuperResolution 的 arg
        if img_lr is None:
            raise ValueError("Conditioning image `img_lr` must be provided.")
        else:
            x = torch.cat((x, img_lr), dim=1)

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        F_x = self.model(
            x.to(dtype),  # (c_in * x).to(dtype),
            torch.zeros(x.shape[0], dtype=dtype, device=x.device),  # c_noise.flatten()
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead.")

        # skip connection
        D_x = F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float, List, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)
