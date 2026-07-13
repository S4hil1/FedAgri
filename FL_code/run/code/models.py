"""
Model architectures for FedAgri.

Includes:
  - Standard backbones (ResNet18, MobileNetV2, EfficientNet, SqueezeNet, …)
  - FedCAMNet (FederatedMobileNetKAN) — paper proposed model
  - Channel / SE attention, KAN layers, ContextAware modules
  - Custom lightweight CNNs (MD2, MiniNetPP, SimpleCNN)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# Attention / KAN building blocks
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1))
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    """Alias for SE-style channel attention (paper terminology)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = SEBlock(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(x)


class FeatureContext(nn.Module):
    """1-D feature gating used in ContextKANHead."""

    def __init__(self, in_features: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_features // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class ContextAwareModule(nn.Module):
    """Spatial feature-map context gating (channel attention on CNN maps)."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        weights = self.fc(self.global_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * weights


class DualPooling(nn.Module):
    """Avg + max pooling fusion."""

    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avg(x) + self.max(x)


class KANLayer(nn.Module):
    """Sinusoidal KAN-style nonlinear layer (primary FedCAMNet head)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x))


# Alias used in some notebooks
KANLayer_111 = KANLayer


class KANLayerRBF(nn.Module):
    """Gaussian RBF-basis KAN layer (ContextKANHead variant)."""

    def __init__(self, in_features: int, out_features: int, num_basis: int = 6):
        super().__init__()
        self.coeff = nn.Parameter(torch.randn(in_features, out_features, num_basis))
        self.register_buffer("grid", torch.linspace(-1, 1, num_basis))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        basis = torch.exp(-((x - self.grid) ** 2))
        return torch.einsum("bin,ion->bo", basis, self.coeff)


class ContextKANHead(nn.Module):
    """FeatureContext → KAN → classifier head."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.context = FeatureContext(in_features)
        self.kan1 = KANLayerRBF(in_features, 128)
        self.kan2 = KANLayerRBF(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context(x)
        x = F.relu(self.kan1(x))
        return self.kan2(x)


# =============================================================================
# FedCAMNet — paper proposed model (FederatedMobileNetKAN)
# =============================================================================

class FederatedMobileNetKAN(nn.Module):
    """
    FedCAMNet: MobileNetV2 backbone + SE channel attention + KAN head.

    Matches FED_MDD_V2 / V3 notebook implementation.
    Set personalization=False to expose a shared `classifier` Linear layer
    (compatible with FedCADR class-wise aggregation on classifier.weight).
    """

    def __init__(self, num_classes: int = 7, personalization: bool = False):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.se = SEBlock(1280)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.kan1 = KANLayer(1280, 1024)
        self.kan2 = KANLayer(1024, 512)
        self.kan3 = KANLayer(512, 256)
        self.kan4 = KANLayer(256, 64)

        self.personalization = personalization
        if personalization:
            self.personal_layer = nn.Linear(64, num_classes)
        else:
            self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.se(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        x = self.kan4(x)
        if self.personalization:
            return self.personal_layer(x)
        return self.classifier(x)


# Paper-facing alias
FedCAMNet = FederatedMobileNetKAN


class FederatedMobileNetKAN_Ablation(nn.Module):
    """Ablation variant: toggle SE and/or KAN components."""

    def __init__(
        self,
        num_classes: int = 7,
        use_se: bool = True,
        use_kan: bool = True,
        personalization: bool = False,
    ):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.use_se = use_se
        self.use_kan = use_kan
        self.personalization = personalization

        if use_se:
            self.se = SEBlock(1280)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if use_kan:
            self.kan1 = KANLayer(1280, 1024)
            self.kan2 = KANLayer(1024, 512)
            self.kan3 = KANLayer(512, 256)
            self.kan4 = KANLayer(256, 64)
            final_dim = 64
        else:
            final_dim = 1280

        if personalization:
            self.personal_layer = nn.Linear(final_dim, num_classes)
        else:
            self.classifier = nn.Linear(final_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.use_se:
            x = self.se(x)
        x = torch.flatten(self.pool(x), 1)
        if self.use_kan:
            x = self.kan4(self.kan3(self.kan2(self.kan1(x))))
        if self.personalization:
            return self.personal_layer(x)
        return self.classifier(x)


class ContextAwareMobileNet(nn.Module):
    """MobileNetV2 + ContextAwareModule + dual pooling classifier."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.context = ContextAwareModule(1280)
        self.pool = DualPooling()
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context(self.features(x))
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(x)


class FinalFLModel(nn.Module):
    """Generic backbone + ContextKANHead (FED_MDD_V3 FinalFLModel)."""

    def __init__(self, model_name: str = "mobilenet", num_classes: int = 7):
        super().__init__()
        self.backbone, out_channels = self._load_backbone(model_name)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = ContextKANHead(out_channels, num_classes)

    def _load_backbone(self, name: str):
        name = name.lower()
        if name in ("mobilenet", "mobilenet_v2"):
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            return base.features, 1280
        if name == "shufflenet":
            base = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            return nn.Sequential(base.conv1, base.maxpool, base.stage2, base.stage3, base.stage4, base.conv5), 1024
        if name == "squeezenet":
            base = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
            return base.features, 512
        if name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            return nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            ), 512
        raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(self.pool(x), 1)
        return self.head(x)


# =============================================================================
# Standard backbones
# =============================================================================

def get_resnet18_model(num_classes: int = 7) -> nn.Module:
    """ResNet18 with 7-class head (main FL backbone in early notebooks)."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_mobilenet_model(num_classes: int = 7) -> nn.Module:
    """MobileNetV2 with replaced classifier."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_efficientnet_model(num_classes: int = 7) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def get_squeezenet_model(num_classes: int = 7) -> nn.Module:
    model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    return model


def get_shufflenet_model(num_classes: int = 7) -> nn.Module:
    model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_densenet_model(num_classes: int = 7) -> nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def get_cnn_model(num_classes: int = 7) -> nn.Module:
    """Legacy CNNModel alias → ResNet18 with correct head."""
    return get_resnet18_model(num_classes)


CNNModel = get_cnn_model


def get_model(name: str, num_classes: int = 7, **kwargs) -> nn.Module:
    """
    Factory for all supported models.

    Names: resnet18, mobilenet, efficientnet, squeezenet, shufflenet,
           densenet, fedcamnet, fedcamnet_ablation, context_aware, md2, mininetpp, simple_cnn
    """
    name = name.lower().replace("-", "_")
    registry = {
        "resnet18": get_resnet18_model,
        "mobilenet": get_mobilenet_model,
        "mobilenet_v2": get_mobilenet_model,
        "efficientnet": get_efficientnet_model,
        "efficientnet_b0": get_efficientnet_model,
        "squeezenet": get_squeezenet_model,
        "shufflenet": get_shufflenet_model,
        "densenet": get_densenet_model,
        "densenet121": get_densenet_model,
        "cnn": get_cnn_model,
        "fedcamnet": lambda nc: FederatedMobileNetKAN(nc, personalization=kwargs.get("personalization", False)),
        "federated_mobilenet_kan": lambda nc: FederatedMobileNetKAN(nc, personalization=kwargs.get("personalization", False)),
        "fedcamnet_ablation": lambda nc: FederatedMobileNetKAN_Ablation(
            nc,
            use_se=kwargs.get("use_se", True),
            use_kan=kwargs.get("use_kan", True),
            personalization=kwargs.get("personalization", False),
        ),
        "context_aware": ContextAwareMobileNet,
        "final_fl": lambda nc: FinalFLModel(kwargs.get("backbone", "mobilenet"), nc),
        "md2": lambda nc: MD2(3, nc),
        "mininetpp": lambda nc: MiniNetPP(3, nc),
        "simple_cnn": SimpleCNN,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {sorted(registry)}")
    return registry[name](num_classes)


# =============================================================================
# Custom lightweight CNNs (MD2 / MiniNetPP / SimpleCNN)
# =============================================================================

class SimpleCNN(nn.Module):
    """Early Maize_Hetero prototype CNN (32×32 input)."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class MD2(nn.Module):
    """
    Lightweight custom CNN with depthwise conv + SE blocks.

    Ported from Maize_disease_classification_pytorch.ipynb.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.dwconv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16)
        self.conv_block1 = nn.Conv2d(16, 32, kernel_size=1)
        self.bn_block1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32, reduction=8)

        self.dwconv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)
        self.conv_block2 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn_block2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64, reduction=8)
        self.residual_proj = nn.Conv2d(32, 64, kernel_size=1)

        self.conv_down = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_down = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128, reduction=8)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(F.silu(self.conv1(x)))

        x = self.se1(self.bn_block1(F.silu(self.conv_block1(self.dwconv1(x)))))

        residual = self.residual_proj(x)
        x = self.se2(self.bn_block2(F.silu(self.conv_block2(self.dwconv2(x)))))
        x = x + residual

        x = self.se3(self.bn_down(F.silu(self.conv_down(x))))
        x = self.global_avg_pool(x).view(x.size(0), -1)
        x = self.dropout(F.silu(self.fc1(x)))
        return self.fc2(x)


class MiniNetPP(nn.Module):
    """
    MiniNet++ — lightweight CNN similar to MD2 with larger DW kernels.

    PyTorch port of the TensorFlow MiniNetPP used in centralized experiments.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.dw1 = nn.Conv2d(16, 16, 5, padding=2, groups=16)
        self.pw1 = nn.Conv2d(16, 32, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32, reduction=8)

        self.dw2 = nn.Conv2d(32, 32, 5, padding=2, groups=32)
        self.pw2 = nn.Conv2d(32, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64, reduction=8)
        self.proj = nn.Conv2d(32, 64, 1)

        self.down = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128, reduction=8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(F.silu(self.conv1(x)))
        x = self.se1(self.bn2(F.silu(self.pw1(self.dw1(x)))))
        residual = self.proj(x)
        x = self.se2(self.bn3(F.silu(self.pw2(self.dw2(x))))) + residual
        x = self.se3(self.bn4(F.silu(self.down(x))))
        return self.fc(self.pool(x).flatten(1))
