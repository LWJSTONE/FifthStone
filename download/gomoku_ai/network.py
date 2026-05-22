"""
轻量化五子棋神经网络 — 深度可分离卷积 + SE注意力
================================================
设计原则:
  1. 极致轻量: 6层残差块 × 64通道，CPU单次推理 <1ms
  2. 深度可分离卷积: 参数量降低 ~8x，推理加速 ~3x
  3. SE(Squeeze-Excitation)注意力: 用极少参数增强特征选择
  4. 双头输出: Policy(策略) + Value(价值)
  5. 权重初始化: He初始化 + BatchNorm稳定训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import (
    BOARD_SIZE, BOARD_SQUARES, INPUT_CHANNELS,
    NUM_RES_BLOCKS, NUM_FILTERS, SE_REDUCTION,
    POLICY_CHANNELS, VALUE_HIDDEN
)


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积: Depthwise + Pointwise，参数量降为标准卷积的 1/K"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise: 每个输入通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels,
            bias=False
        )
        # Pointwise: 1x1卷积融合通道
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-Excitation 注意力模块"""

    def __init__(self, channels, reduction):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """残差块: 深度可分离卷积 + SE注意力 + 跳跃连接"""

    def __init__(self, channels, reduction):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, 3, padding=1)
        self.conv2 = DepthwiseSeparableConv(channels, channels, 3, padding=1)
        self.se = SEBlock(channels, reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        return out


class PolicyHead(nn.Module):
    """策略头: 输出每个位置的落子概率"""

    def __init__(self, in_channels, policy_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, policy_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(policy_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(policy_channels * BOARD_SIZE * BOARD_SIZE, BOARD_SQUARES)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # 返回logits(外部做softmax + mask)


class ValueHead(nn.Module):
    """价值头: 输出局面评估值 [-1, 1]"""

    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x.squeeze(-1)


class GomokuNet(nn.Module):
    """
    五子棋双头神经网络
    ==================
    输入: (batch, 17, 15, 15) 特征平面
    输出: (policy_logits, value)
      - policy_logits: (batch, 225) 策略logits
      - value: (batch,) 局面价值 [-1, 1]
    """

    def __init__(self):
        super().__init__()
        # 输入卷积
        self.input_conv = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, NUM_FILTERS, 3, padding=1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(inplace=True)
        )

        # 残差塔
        self.res_blocks = nn.ModuleList([
            ResBlock(NUM_FILTERS, SE_REDUCTION)
            for _ in range(NUM_RES_BLOCKS)
        ])

        # 双头
        self.policy_head = PolicyHead(NUM_FILTERS, POLICY_CHANNELS)
        self.value_head = ValueHead(NUM_FILTERS, VALUE_HIDDEN)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """He初始化 + BatchNorm权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """前向传播"""
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def predict(self, feature_planes, legal_mask=None):
        """
        单样本推理接口(用于MCTS)
        ========================
        参数:
          feature_planes: (17, 15, 15) numpy数组
          legal_mask: (225,) bool数组，合法着法掩码
        返回:
          policy: (225,) numpy数组，合法着法概率分布
          value: float，局面评估值
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature_planes).float().unsqueeze(0)
            policy_logits, value = self.forward(x)

            # 合法着法掩码
            if legal_mask is not None:
                mask = torch.from_numpy(legal_mask).float()
                policy_logits = policy_logits.squeeze(0) + (1 - mask) * (-1e9)
            else:
                policy_logits = policy_logits.squeeze(0)

            policy = F.softmax(policy_logits, dim=0).numpy()
            value = value.item()

        return policy, value

    def predict_batch(self, feature_planes_list, legal_masks=None):
        """
        批量推理接口(用于并行MCTS)
        =========================
        参数:
          feature_planes_list: list of (17, 15, 15) numpy数组
          legal_masks: list of (225,) numpy数组，或None
        返回:
          policies: (batch, 225) numpy数组
          values: (batch,) numpy数组
        """
        self.eval()
        batch_size = len(feature_planes_list)
        with torch.no_grad():
            x = np.stack(feature_planes_list)
            x = torch.from_numpy(x).float()
            policy_logits, values = self.forward(x)
            policy_logits = policy_logits.numpy()
            values = values.numpy()

            # 合法着法掩码
            if legal_masks is not None:
                masks = np.stack(legal_masks).astype(np.float32)
                policy_logits = policy_logits + (1 - masks) * (-1e9)

            # 批量softmax
            exp_logits = np.exp(policy_logits - np.max(policy_logits, axis=1, keepdims=True))
            policies = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return policies, values


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_model(device='cpu'):
    """创建模型并移至指定设备"""
    model = GomokuNet().to(device)
    total, trainable = count_parameters(model)
    print(f"[Network] 模型参数量: 总计={total:,} 可训练={trainable:,}")
    print(f"[Network] 架构: {NUM_RES_BLOCKS}残差块 × {NUM_FILTERS}通道 + "
          f"SE(r={SE_REDUCTION}) + 深度可分离卷积")
    return model
