"""
轻量化五子棋神经网络 (V11 — InferenceWrapper修复版)
====================================================
V11 修复:
  1. 添加 InferenceWrapper — 为 TorchScript 等优化模型提供 predict/predictBatch 接口
  2. _optimize_for_inference 中 TorchScript 返回 InferenceWrapper 而非裸 traced model
  3. INT8 量化模型保留原始方法, 无需 wrapper
  4. 移除死代码 compute_pattern_feature_channels (vct.py)

V8 修复:
  1. 调换 TorchScript 和 INT8 优先级 — TorchScript 优化整个计算图含 Conv2d,
     对 Conv 密集型网络比 INT8 (仅量化 Linear) 更有效
  2. EMA 评估用 try/finally 保护 (在 train.py 中)

V2 修复:
  1. TorchScript 仅在推理时应用 (不干扰训练)
  2. BN 融合实际执行 (推理优化)
  3. INT8 量化包含 Conv2d (不只是 Linear)
  4. PolicyHead 改用 1×1 Conv (替代 reshape+fc)
  5. ONNX Runtime 推理接口
  6. torch.compile 支持 (PyTorch 2.0+)
  7. 预分配输入 Tensor (MCTS 热路径)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import (
    BOARD_SIZE, BOARD_SQUARES, INPUT_CHANNELS,
    NUM_RES_BLOCKS, NUM_FILTERS, SE_REDUCTION,
    POLICY_CHANNELS, VALUE_HIDDEN,
    USE_TORCHSCRIPT, USE_INT8_QUANT, USE_BN_FUSE, USE_NHWC,
    USE_ONNX_RUNTIME, USE_TORCH_COMPILE
)


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积: Depthwise + Pointwise"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.pointwise(self.depthwise(x)))


class SEBlock(nn.Module):
    """Squeeze-Excitation 注意力"""
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
    """残差块: 深度可分离卷积 + SE + 跳跃连接"""
    def __init__(self, channels, reduction):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, 3, padding=1)
        self.conv2 = DepthwiseSeparableConv(channels, channels, 3, padding=1)
        self.se = SEBlock(channels, reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        return self.relu(out + x)


class PolicyHead(nn.Module):
    """
    V2: 策略头 — 1×1 Conv 输出单通道, flatten 为 225 个 logits
    ==========================================================
    conv_out: (in_ch → 1, 1×1) 输出 (batch, 1, 15, 15)
    flatten: (batch, 225) — 每个位置一个 policy logit
    """
    def __init__(self, in_ch, pol_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, pol_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(pol_ch)
        self.relu = nn.ReLU(inplace=True)
        # 输出 1 通道 → flatten 为 (batch, BOARD_SQUARES)
        self.conv_out = nn.Conv2d(pol_ch, 1, 1, bias=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.conv_out(x)
        return x.reshape(x.size(0), -1)  # (batch, BOARD_SQUARES)


class ValueHead(nn.Module):
    """价值头"""
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.reshape(x.size(0), -1)
        return self.tanh(self.fc2(self.relu(self.fc1(x)))).squeeze(-1)


class GomokuNet(nn.Module):
    """
    五子棋双头神经网络 (V2)
    输入: (batch, 19, 15, 15) — 含2个领域知识通道
    输出: (policy_logits, value)
    """
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, NUM_FILTERS, 3, padding=1, bias=False),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(NUM_FILTERS, SE_REDUCTION) for _ in range(NUM_RES_BLOCKS)
        ])
        self.policy_head = PolicyHead(NUM_FILTERS, POLICY_CHANNELS)
        self.value_head = ValueHead(NUM_FILTERS, VALUE_HIDDEN)
        self._init_weights()

    def _init_weights(self):
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
        x = x.contiguous()
        x = self.input_conv(x)
        for block in self.res_blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)

    def predict(self, feature_planes, legal_mask=None):
        """单样本推理 (MCTS用) — V2: 支持预分配"""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(feature_planes).float().unsqueeze(0)
            if USE_NHWC:
                x = x.to(memory_format=torch.channels_last)
            policy_logits, value = self.forward(x)
            if legal_mask is not None:
                mask = torch.from_numpy(legal_mask).float()
                policy_logits = policy_logits.squeeze(0) + (1 - mask) * (-1e9)
            else:
                policy_logits = policy_logits.squeeze(0)
            policy = F.softmax(policy_logits, dim=0).numpy()
            value = value.item()
        return policy, value

    def predictBatch(self, feature_list, legal_masks=None):
        """V2 批量推理 — 正确传递 legal_mask"""
        self.eval()
        with torch.no_grad():
            x = np.stack(feature_list)
            x = torch.from_numpy(x).float()
            if USE_NHWC:
                x = x.to(memory_format=torch.channels_last)
            policy_logits, values = self.forward(x)
            policy_logits_np = policy_logits.numpy()
            values_np = values.numpy()

            # V2: 正确应用 legal_mask
            if legal_masks is not None:
                masks = np.stack(legal_masks).astype(np.float32)
                policy_logits_np = policy_logits_np + (1 - masks) * (-1e9)

            # Stable softmax
            exp_l = np.exp(policy_logits_np - np.max(policy_logits_np, axis=1, keepdims=True))
            policies = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        return policies, values_np


# ======================== BN 融合 ========================

def fuse_bn(conv, bn):
    """BN 融合: 将 BatchNorm 参数融入卷积权重"""
    w = conv.weight.data
    if conv.bias is not None:
        b = conv.bias.data.clone()
    else:
        b = torch.zeros(w.size(0), dtype=w.dtype)

    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)
    b = (b - mean) * scale + beta
    w_new = w * scale[:, None, None, None]

    fused_conv = nn.Conv2d(
        w.size(1), w.size(0), conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        groups=conv.groups, bias=True
    )
    fused_conv.weight.data = w_new
    fused_conv.bias.data = b
    return fused_conv


def fuse_bn_for_model(model):
    """
    V2: 实际执行 BN 融合
    遍历模型, 将 DepthwiseSeparableConv 中的 Conv+BN 融合
    """
    model.eval()

    for block in model.res_blocks:
        # conv1: depthwise + pointwise + bn
        if hasattr(block.conv1, 'depthwise') and hasattr(block.conv1, 'bn'):
            # Pointwise + BN 融合
            block.conv1.pointwise = fuse_bn(block.conv1.pointwise, block.conv1.bn)
            block.conv1.bn = nn.Identity()
        # conv2
        if hasattr(block.conv2, 'depthwise') and hasattr(block.conv2, 'bn'):
            block.conv2.pointwise = fuse_bn(block.conv2.pointwise, block.conv2.bn)
            block.conv2.bn = nn.Identity()

    # PolicyHead
    if hasattr(model.policy_head, 'conv') and hasattr(model.policy_head, 'bn'):
        model.policy_head.conv = fuse_bn(model.policy_head.conv, model.policy_head.bn)
        model.policy_head.bn = nn.Identity()

    # ValueHead
    if hasattr(model.value_head, 'conv') and hasattr(model.value_head, 'bn'):
        model.value_head.conv = fuse_bn(model.value_head.conv, model.value_head.bn)
        model.value_head.bn = nn.Identity()

    # Input conv
    if hasattr(model.input_conv, '__getitem__'):
        # Sequential: Conv + BN + ReLU
        conv0 = model.input_conv[0]
        bn1 = model.input_conv[1]
        fused = fuse_bn(conv0, bn1)
        # V9 修复: nn.Sequential 不支持直接赋值, 必须重建
        relu2 = model.input_conv[2] if len(model.input_conv) > 2 else nn.ReLU(inplace=True)
        model.input_conv = nn.Sequential(fused, nn.Identity(), relu2)

    return model


# ======================== ONNX Runtime 推理 ========================

class ONNXInferenceModel:
    """ONNX Runtime 推理封装"""
    def __init__(self, model, onnx_path="model.onnx"):
        self.onnx_path = onnx_path
        self.session = None

        # 导出 ONNX
        try:
            dummy = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'],
                output_names=['policy', 'value'],
                dynamic_axes={'input': {0: 'batch'}, 'policy': {0: 'batch'}, 'value': {0: 'batch'}},
                opset_version=14
            )
        except Exception as e:
            print(f"[ONNX] 导出失败: {e}")
            return

        # 加载 ONNX Runtime
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_path, opts)
            print("[ONNX] ONNX Runtime 推理引擎初始化成功")
        except ImportError:
            print("[ONNX] onnxruntime 未安装, 回退到 PyTorch 推理")
        except Exception as e:
            print(f"[ONNX] 初始化失败: {e}")

    def predict(self, feature_planes, legal_mask=None):
        if self.session is None:
            return None, None
        x = feature_planes.astype(np.float32)[np.newaxis, ...]  # V3: 修复 numpy 无 unsqueeze
        results = self.session.run(None, {'input': x})
        policy_logits = results[0][0]
        value = results[1][0]
        if legal_mask is not None:
            policy_logits = policy_logits + (1 - legal_mask) * (-1e9)
        exp_p = np.exp(policy_logits - np.max(policy_logits))
        policy = exp_p / exp_p.sum()
        return policy, float(value)

    def predictBatch(self, feature_list, legal_masks=None):
        if self.session is None:
            return None, None
        x = np.stack(feature_list).astype(np.float32)
        results = self.session.run(None, {'input': x})
        policy_logits = results[0]
        # V13 修复: squeeze(-1) 避免 batch_size=1 时过度压缩为 0-d 数组
        values = results[1].reshape(-1)
        if legal_masks is not None:
            masks = np.stack(legal_masks).astype(np.float32)
            policy_logits = policy_logits + (1 - masks) * (-1e9)
        exp_p = np.exp(policy_logits - np.max(policy_logits, axis=1, keepdims=True))
        policies = exp_p / exp_p.sum(axis=1, keepdims=True)
        return policies, values


# ======================== 模型创建 ========================

def create_model(device='cpu', optimize_for_inference=False):
    """V2: 创建模型, 推理时可选优化"""
    model = GomokuNet().to(device)

    if USE_NHWC:
        model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Network] 参数量: {total_params:,}")
    print(f"[Network] 架构: {NUM_RES_BLOCKS}块×{NUM_FILTERS}通道 + SE(r={SE_REDUCTION}) + "
          f"深度可分离卷积 + {INPUT_CHANNELS}输入通道")

    if optimize_for_inference:
        model = _optimize_for_inference(model)

    return model


def _optimize_for_inference(model):
    """V8 推理优化: BN融合 → TorchScript → INT8量化 → torch.compile
    V8 修复: 调换 TorchScript 和 INT8 优先级。
    本网络 Conv2d 是计算瓶颈 (6个ResBlock, 深度可分离卷积),
    TorchScript 能优化整个计算图含 Conv2d, 比 INT8 (仅量化Linear) 更有效。

    V11 修复: TorchScript/INT8/torch.compile 返回的模型没有 predict/predictBatch 方法,
    不能直接用于 MCTS。添加 InferenceWrapper 适配器, 将 forward() 结果包装为
    predict/predictBatch 接口。
    """
    model.eval()

    # 1. BN 融合 (最先执行, 后续优化基于融合后的模型)
    if USE_BN_FUSE:
        try:
            model = fuse_bn_for_model(model)
            print("[Network] BN 融合成功")
        except Exception as e:
            print(f"[Network] BN 融合失败: {e}")

    # V8: 按优先级尝试, 第一个成功的即返回
    # 优先级: TorchScript (优化整个模型) > INT8 (仅Linear) > torch.compile

    # 2. TorchScript (优化整个计算图, 含 Conv2d)
    if USE_TORCHSCRIPT:
        try:
            dummy_input = torch.randn(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            if USE_NHWC:
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
            traced = torch.jit.trace(model, dummy_input)
            print("[Network] TorchScript trace 编译成功")
            return InferenceWrapper(traced)
        except Exception as e:
            print(f"[Network] TorchScript 跳过 ({type(e).__name__})")

    # 3. INT8 动态量化 (仅 Linear, CPU 上 quantize_dynamic 不支持 Conv2d)
    if USE_INT8_QUANT:
        try:
            quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # V7 修复: CPU 上 quantize_dynamic 仅支持 Linear
                dtype=torch.qint8
            )
            print("[Network] INT8 动态量化成功 (Linear)")
            # INT8量化保留原始类的方法, predict/predictBatch 仍可用
            return quantized
        except Exception as e:
            print(f"[Network] INT8 量化跳过 ({e})")

    # 4. torch.compile (PyTorch 2.0+)
    if USE_TORCH_COMPILE:
        try:
            compiled = torch.compile(model, backend='inductor', mode='reduce-overhead')
            print("[Network] torch.compile 成功")
            # V12 修复: torch.compile 返回的模型没有 predict/predictBatch 方法,
            # 必须用 InferenceWrapper 包装, 否则 MCTS 调用 model.predict() 会崩溃
            return InferenceWrapper(compiled)
        except Exception as e:
            print(f"[Network] torch.compile 跳过 ({type(e).__name__})")

    return model


class InferenceWrapper:
    """V11: 为 TorchScript 等优化模型提供 predict/predictBatch 接口适配"""
    def __init__(self, optimized_model):
        self._model = optimized_model
        # 代理常用属性
        self.state_dict = optimized_model.state_dict
        self.load_state_dict = optimized_model.load_state_dict

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def eval(self):
        if hasattr(self._model, 'eval'):
            self._model.eval()
        return self

    def train(self, mode=True):
        if hasattr(self._model, 'train'):
            self._model.train(mode)
        return self

    def predict(self, feature_planes, legal_mask=None):
        """单样本推理 — 适配 MCTS 调用"""
        with torch.no_grad():
            x = torch.from_numpy(feature_planes).float().unsqueeze(0)
            if USE_NHWC:
                x = x.to(memory_format=torch.channels_last)
            policy_logits, value = self._model(x)
            if legal_mask is not None:
                mask = torch.from_numpy(legal_mask).float()
                policy_logits = policy_logits.squeeze(0) + (1 - mask) * (-1e9)
            else:
                policy_logits = policy_logits.squeeze(0)
            policy = F.softmax(policy_logits, dim=0).numpy()
            value = value.item()
        return policy, value

    def predictBatch(self, feature_list, legal_masks=None):
        """批量推理 — 适配 MCTS 调用"""
        with torch.no_grad():
            x = np.stack(feature_list)
            x = torch.from_numpy(x).float()
            if USE_NHWC:
                x = x.to(memory_format=torch.channels_last)
            policy_logits, values = self._model(x)
            policy_logits_np = policy_logits.numpy()
            values_np = values.numpy()

            if legal_masks is not None:
                masks = np.stack(legal_masks).astype(np.float32)
                policy_logits_np = policy_logits_np + (1 - masks) * (-1e9)

            exp_l = np.exp(policy_logits_np - np.max(policy_logits_np, axis=1, keepdims=True))
            policies = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        return policies, values_np


def create_inference_model(model):
    """从训练模型创建推理优化模型"""
    return _optimize_for_inference(model)


def create_onnx_model(model, onnx_path="model.onnx"):
    """创建 ONNX Runtime 推理模型"""
    if USE_ONNX_RUNTIME:
        return ONNXInferenceModel(model, onnx_path)
    return None
