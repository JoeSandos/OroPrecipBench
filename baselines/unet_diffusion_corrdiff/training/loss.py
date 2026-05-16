# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch import Tensor
from torch_utils import persistence
from typing import Any, List, Literal, Tuple, Union, Set, Dict, Optional, Callable

from abc import ABC, abstractmethod

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        # 对应 EDM 论文中的 ln(σ) = N(P_mean, P_std^2)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # 是否进行数据增强
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma  # 噪声 noise
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Variation of the loss function proposed in the EDM paper for Super-Resolution.


@persistence.persistent_class
class EDMLossSR:
    """
    Variation of the loss function proposed in the EDM paper for Super-Resolution.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'P_mean' and 'P_std' random values. The calculated loss is weighted as a
        function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        # 对应 EDM 论文中的 ln(σ) = N(P_mean, P_std^2)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generation
        # 是否进行数据增强
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        n = torch.randn_like(y) * sigma  # 噪声 noise
        D_yn = net(y + n, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 回归损失函数，用于确定性预测。


@persistence.persistent_class
class RegressionLoss:
    """
    Regression loss function for the deterministic predictions.
    Note: this loss does not apply any reduction.

    Attributes
    ----------
    sigma_data: float
        Standard deviation for data. Deprecated and ignored.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self):
        """
        Arguments
        ----------
        """
        return

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
        augment_pipe=None,
        lead_time_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate and return the regression loss for deterministic predictions.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model that will make predictions.
            Expected signature: `net(x, img_lr,
            augment_labels=augment_labels, force_fp32=False)`, where:
                x (torch.Tensor): Tensor of shape (B, C_hr, H, W). Is zero-filled.
                img_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                augment_labels (torch.Tensor, optional): Optional augmentation
                labels, returned by `augment_pipe`.
                force_fp32 (bool, optional): Whether to force the model to use
                fp32, by default False.
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        img_clean : torch.Tensor
            High-resolution input images of shape (B, C_hr, H, W).
            Used as ground truth and for data augmentation if 'augment_pipe' is provided.

        img_lr : torch.Tensor
            Low-resolution input images of shape (B, C_lr, H, W).
            Used as input to the neural network.

        augment_pipe : callable, optional
            An optional data augmentation function.
            Expected signature:
                img_tot (torch.Tensor): Concatenated high and low resolution
                    images of shape (B, C_hr+C_lr, H, W)
            Returns:
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - Augmented images of shape (B, C_hr+C_lr, H, W)
                    - Optional augmentation labels

        lead_time_label : Optional[torch.Tensor], optional
            Lead time labels for temporal predictions, by default None.
            Shape can vary based on model requirements, typically (B,) or scalar.

        Returns
        -------
        torch.Tensor
            A tensor representing the per-sample element-wise squared
            difference between the network's predictions and the high
            resolution images `img_clean` (possibly data-augmented by
            `augment_pipe`).
            Shape: (B, C_hr, H, W), same as `img_clean`.
        """
        weight = 1.0  # (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # 是否进行数据增强
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        # 对应 EDMLossSR 的 y + n
        zero_input = torch.zeros_like(y, device=img_clean.device)

        if lead_time_label is not None:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                zero_input,
                y_lr,
                force_fp32=False,
                augment_labels=augment_labels,
            )

        loss = weight * ((D_yn - y) ** 2)

        return loss


# ----------------------------------------------------------------------------
# 残差损失


@persistence.persistent_class
class ResidualLoss:
    """
    Mixture loss function for denoising score matching.

    This class implements a loss function that combines deterministic
    regression with denoising score matching. It uses a pre-trained regression
    network to compute residuals before applying the diffusion process.

    Parameters
    ----------
    regression_net : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.
        指示是否使用高分辨率均值作为 condition.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric
    Downscaling. arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
    ):
        """
        Arguments
        ----------
        regression_net : torch.nn.Module
            Pre-trained regression network used to compute residuals.
            Expected signature: `net(zero_input, y_lr,
            lead_time_label=lead_time_label, augment_labels=augment_labels)` or
            `net(zero_input, y_lr, augment_labels=augment_labels)`, where:
                zero_input (torch.Tensor): Zero tensor of shape (B, C_hr, H, W)
                y_lr (torch.Tensor): Low-resolution input of shape (B, C_lr, H, W)
                lead_time_label (torch.Tensor, optional): Optional lead time labels
                augment_labels (torch.Tensor, optional): Optional augmentation labels
            Returns:
                torch.Tensor: Predictions of shape (B, C_hr, H, W)

        P_mean : float, optional
            Mean value for noise level computation, by default 0.0.

        P_std : float, optional
            Standard deviation for noise level computation, by default 1.2.

        sigma_data : float, optional
            Standard deviation for data weighting, by default 0.5.

        hr_mean_conditioning : bool, optional
            Whether to use high-resolution mean for conditioning predicted, by default False.
            When True, the mean prediction from `regression_net` is channel-wise concatenated with `img_lr` for conditioning.
            是否使用高分辨率均值进行条件预测，默认为False。
            当为 True 时，来自 `regression_net` 的平均预测与 `img_lr` 按通道连接以进行调节。
        """
        self.regression_net = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hr_mean_conditioning = hr_mean_conditioning
        self.y_mean = None

    def get_noise_params(self, y: Tensor) -> Tensor:
        """
        Compute the noise parameters to apply denoising score matching.

        Parameters
        ----------
        y : torch.Tensor
            Latent state of shape :math:`(B, *)`. Only used to determine the shape of
            the noise and create tensors on the same device.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Noise ``n`` of shape :math:`(B, *)` to be added to the latent state.
            - Noise level ``sigma`` of shape :math:`(B, 1, 1, 1)`.
            - Weight ``weight`` of shape :math:`(B, 1, 1, 1)` to multiply the loss.
        """
        # Sample noise level
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample noise
        n = torch.randn_like(y) * sigma
        return n, sigma, weight

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching=None,  # 是否使用 分块
        lead_time_label: Optional[Tensor] = None,
        labels=None,
        augment_pipe=None,
        use_patch_grad_acc: bool = False,
    ) -> Tensor:
        """
        计算并返回去噪分数匹配（denoising score matching）的损失。

        此方法计算混合损失，结合了确定性回归（deterministic regression）和去噪分数匹配。
        它首先使用回归网络计算残差，然后将扩散过程应用于这些残差。

        除了标准的去噪分数匹配损失外，此方法还支持用于多重扩散（multi-diffusion）的可选分块（patching）。
        在这种情况下，输入的空间维度被分解为 `P` 个形状为 (H_patch, W_patch) 的较小块，
        这些块沿批次维度分组，并且模型单独应用于每个块。
        在下文中，如果未提供 `patching`，则不对输入进行分块，即 `P=1` 且 `(H_patch, W_patch) = (H, W)`。
        当使用分块时，原始未分块的调节信息（conditioning）被插值到形状为 `(H_patch, W_patch)` 的空间网格上，
        并按通道连接到分块的调节信息。这确保了每个块都保留了来自整个域的全局信息。

        扩散模型 `net` 期望以具有 `C_cond` 个通道的输入为条件，其中：
            - 如果 `hr_mean_conditioning` 为 `False` 且 `patching` 为 None，则 `C_cond = C_lr`。
            - 如果 `hr_mean_conditioning` 为 `True` 且 `patching` 为 None，则 `C_cond = C_hr + C_lr`。
            - 如果 `hr_mean_conditioning` 为 `True` 且 `patching` 不为 None，则 `C_cond = C_hr + 2*C_lr`。
            - 如果 `hr_mean_conditioning` 为 `False` 且 `patching` 不为 None，则 `C_cond = 2*C_lr`。
        此外，`C_cond` 还应包括任何嵌入通道，例如位置嵌入或时间嵌入。

        注意：此损失函数不应用任何归约（reduction）。

        Parameters
        ----------
        net : torch.nn.Module
            用于扩散过程的神经网络模型。
            预期签名：`net(latent, y_lr, sigma, embedding_selector=embedding_selector, lead_time_label=lead_time_label, augment_labels=augment_labels)`，其中：
                latent (torch.Tensor): 噪声输入，形状为 (B[*P], C_hr, H_patch, W_patch)
                y_lr (torch.Tensor): 调节信息，形状为 (B[*P], C_cond, H_patch, W_patch)
                sigma (torch.Tensor): 噪声水平，形状为 (B[*P], 1, 1, 1)
                embedding_selector (callable, optional): 选择位置嵌入的函数。仅在提供 `patching` 时使用。
                lead_time_label (torch.Tensor, optional): 提前期标签。
                augment_labels (torch.Tensor, optional): 增强标签。
            返回：
                torch.Tensor: 预测结果，形状为 (B[*P], C_hr, H_patch, W_patch)

        img_clean : torch.Tensor
            高分辨率输入图像，形状为 (B, C_hr, H, W)。
            用作地面实况（ground truth），如果提供了 `augment_pipe`，则用于数据增强。

        img_lr : torch.Tensor
            低分辨率输入图像，形状为 (B, C_lr, H, W)。
            用作回归网络的输入和扩散过程的调节信息。

        patching : Optional[RandomPatching2D], optional
            用于处理大图像的分块策略，默认为 None。详情请参阅 :class:`physicsnemo.utils.patching.RandomPatching2D`。
            当提供时，分块策略用于扩散模型 `net` 中的图像块和位置嵌入选择。
            将张量从形状 (B, C, H, W) 转换为 (B*P, C, H_patch, W_patch)。

        lead_time_label : Optional[torch.Tensor], optional
            用于提前期感知预测的标签，默认为 None。
            形状可能因模型要求而异，通常为 (B,) 或标量。

        augment_pipe : Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]]
            数据增强函数。
            预期签名：
                img_tot (torch.Tensor): 连接的高分辨率和低分辨率图像，形状为 (B, C_hr+C_lr, H, W)
            返回：
                Tuple[torch.Tensor, Optional[torch.Tensor]]:
                    - 增强后的图像，形状为 (B, C_hr+C_lr, H, W)
                    - 可选的增强标签
        use_patch_grad_acc: bool, optional
            布尔标志，指示是否启用多重迭代的分块累积以分摊回归成本。默认为 False。

        Returns
        -------
        torch.Tensor
            如果不使用分块：
                形状为 (B, C_hr, H, W) 的张量，表示每个样本的损失。
            如果使用分块：
                形状为 (B*P, C_hr, H_patch, W_patch) 的张量，表示每个块的损失。

        Raises
        ------
        ValueError
            如果提供了 patching 但不是 RandomPatching2D 的实例。
            如果 img_clean 和 img_lr 的形状不兼容。
        """

        # 不使用 patching
        if patching:
            raise NotImplementedError("Patching is not implemented yet in ResidualLoss.")

        # 不使用 use_patch_grad_acc
        if use_patch_grad_acc:
            raise NotImplementedError("use_patch_grad_acc is not implemented yet in ResidualLoss.")

        # 不使用 lead_time_label
        if lead_time_label:
            raise NotImplementedError("lead_time_label is not implemented yet in ResidualLoss.")

        # 安全检查：强制形状匹配
        if img_clean.shape[0] != img_lr.shape[0] or img_clean.shape[2:] != img_lr.shape[2:]:
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        # 为条件生成进行增强
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:, : img_clean.shape[1], :, :]  # img_clean
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]  # img_lr
        y_lr_res = y_lr  # img_lr
        batch_size = y.shape[0]  # 只有 patching is not None 时, 才有用

        """
        # 计算高分辨率均值 y_mean
        # 这里的 regression_net 是 CorrDiffRegressionUNet, 它的 forward 需要:
            - zero_input: [B, C_hr, H, W]
            - y_lr: [B, C_lr, H, W]
        """
        self.y_mean = self.regression_net(
            torch.zeros_like(y, device=img_clean.device),
            y_lr_res,
            augment_labels=augment_labels,
        )

        y = y - self.y_mean  # 残差

        if self.hr_mean_conditioning:
            # 将 y_mean 与 y_lr 按通道连接
            y_lr = torch.cat((self.y_mean, y_lr), dim=1)

            ########### 测试代码 START ###########
            # 这里用 y 代替 y_mean 进行拼接.
            # y_lr = torch.cat((y, y_lr), dim=1)
            # 结论: 用 y 代替 y_mean 进行拼接, 是work的
            ########### 测试代码 END ###########

        # 采样噪声水平
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        # 对应 EDM 论文中的 ln(σ) = N(P_mean, P_std^2)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # 损失权重
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(y) * sigma  # 噪声 noise

        D_yn = net(y + n, y_lr, sigma, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)

        return loss
