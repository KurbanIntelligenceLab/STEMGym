from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):


    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):


    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Up(nn.Module):


    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class AtomFinderUNet(nn.Module):


    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.inc = _DoubleConv(in_channels, c)
        self.down1 = _Down(c, c * 2)
        self.down2 = _Down(c * 2, c * 4)
        self.down3 = _Down(c * 4, c * 8)
        self.up1 = _Up(c * 8, c * 4)
        self.up2 = _Up(c * 4, c * 2)
        self.up3 = _Up(c * 2, c)
        self.outc = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return torch.sigmoid(self.outc(x))


class AtomFinderEnsemble(nn.Module):


    def __init__(self, n_members: int = 3, **unet_kwargs):
        super().__init__()
        self.members = nn.ModuleList(
            [AtomFinderUNet(**unet_kwargs) for _ in range(n_members)]
        )
        self._no_uncertainty = False

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all ensemble members."""
        if self._no_uncertainty:
            pred = self.members[0](x)
            return pred, torch.zeros_like(pred)
        preds = torch.stack([m(x) for m in self.members], dim=0)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std

    def predict_single(
        self, x: torch.Tensor, member_idx: int = 0
    ) -> torch.Tensor:
        """Predict with a single ensemble member (for training)."""
        return self.members[member_idx](x)


class DefectClassifierCNN(nn.Module):


    def __init__(self, n_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_forward: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout inference for uncertainty estimation."""
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_forward):
                logits = self.forward(x)
                preds.append(F.softmax(logits, dim=-1))
        self.eval()

        preds = torch.stack(preds, dim=0)
        mean_probs = preds.mean(dim=0)
        entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=-1)
        return mean_probs, entropy


class _ResBlock(nn.Module):


    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class PhaseIdentifierResNet(nn.Module):


    def __init__(self, n_phases: int = 2, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = _ResBlock(c)
        self.layer2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            _ResBlock(c * 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            _ResBlock(c * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c * 4, n_phases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
