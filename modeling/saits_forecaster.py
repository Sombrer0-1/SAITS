"""
SAITS Forecaster - 基于 SAITS 架构的时间序列预测模型

设计思想:
1. 复用 SAITS 的 Self-Attention 编码器 (已验证的时序建模能力)
2. 新增预测解码器 (Decoder) 或预测头 (Projection Head)
3. 支持两阶段训练: 历史重建 + 未来预测 (继承 SAITS 的联合优化思想)

与 SAITS 的关系:
- SAITS: Encoder → Imputed Data (填补任务)
- SAITS_Forecaster: Encoder → Decoder → Forecasted Data (预测任务)

Created for SAITS Forecasting Extension
License: MIT
"""

import torch
import torch.nn as nn
from modeling.layers import EncoderLayer, PositionalEncoding
from modeling.saits import SAITS


class ForecastProjectionHead(nn.Module):
    """
    简单的预测头
    
    将编码器输出投影到未来时间步
    适用于: 快速原型,短期预测
    
    架构: 
        方案1 - 使用全局表示 (默认):
            encoded_history [batch, history_len, d_model]
            → mean pooling → [batch, d_model]
            → FC layers → [batch, forecast_horizon * feature_num]
            → reshape → [batch, forecast_horizon, feature_num]
        
        方案2 - 逐步投影:
            encoded_history [batch, history_len, d_model]
            → 逐时间步投影 → [batch, history_len, feature_num]
            → 时间维度变换 → [batch, forecast_horizon, feature_num]
    """
    def __init__(self, d_model, history_len, forecast_horizon, feature_num, dropout=0.1):
        super().__init__()
        self.history_len = history_len
        self.forecast_horizon = forecast_horizon
        self.feature_num = feature_num
        self.d_model = d_model
        
        # 方案1: 全局表示 → 预测未来 (简单有效)
        # 使用平均池化聚合所有历史编码
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, forecast_horizon * feature_num)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_model * 2)
    
    def forward(self, encoded_history):
        """
        参数:
            encoded_history: [batch, history_len, d_model]
        
        返回:
            forecast: [batch, forecast_horizon, feature_num]
        """
        batch_size = encoded_history.size(0)
        
        # 聚合所有历史编码 (平均池化)
        # 这样可以利用整个历史窗口的信息,而不只是最后一步
        global_repr = torch.mean(encoded_history, dim=1)  # [batch, d_model]
        
        # 投影到未来
        x = self.fc1(global_repr)  # [batch, d_model * 2]
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch, forecast_horizon * feature_num]
        
        # 重塑为时间序列
        forecast = x.view(batch_size, self.forecast_horizon, self.feature_num)
        
        return forecast


class SAITS_Forecaster(nn.Module):
    """
    基于 SAITS 的预测模型
    
    架构:
        历史数据 → [SAITS Encoder] → 编码表示 → [Projection Head] → 未来预测
                          ↑ 复用                          ↑ 新增
    
    优势:
    1. 复用 SAITS 的编码器 (对角掩码, 多头注意力等)
    2. 支持处理历史数据中的缺失值
    3. 轻量级,易于训练
    """
    def __init__(
        self,
        n_groups,
        n_group_inner_layers,
        history_len,
        forecast_horizon,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        device,
        **kwargs
    ):
        super().__init__()
        self.history_len = history_len
        self.forecast_horizon = forecast_horizon
        self.d_feature = d_feature
        self.device = device
        
        # ====== 复用 SAITS 编码器 ======
        # 注意: 这里 d_time 设置为 history_len
        self.saits_encoder = SAITS(
            n_groups=n_groups,
            n_group_inner_layers=n_group_inner_layers,
            d_time=history_len,  # 关键: 使用历史长度
            d_feature=d_feature,
            d_model=d_model,
            d_inner=d_inner,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            input_with_mask=kwargs.get('input_with_mask', True),
            MIT=False,  # 预测任务不需要 MIT
            diagonal_attention_mask=kwargs.get('diagonal_attention_mask', True),
            param_sharing_strategy=kwargs.get('param_sharing_strategy', 'inner_group')
        )
        
        # ====== 新增: 预测头 ======
        self.forecast_head = ForecastProjectionHead(
            d_model=d_model,
            history_len=history_len,
            forecast_horizon=forecast_horizon,
            feature_num=d_feature,
            dropout=dropout
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
    
    def encode_history(self, history, missing_mask=None):
        """
        使用 SAITS 编码器编码历史数据
        
        关键机制: 利用 missing_mask 告知模型哪些是真实观测,哪些是填充值
        - history 中缺失位置已填充为 0
        - missing_mask 标记: 1=真实观测, 0=缺失填充
        - 通过 concat([history, missing_mask]) 让模型区分真实值和填充值
        
        参数:
            history: [batch, history_len, feature_num] - 缺失位置填充为 0
            missing_mask: [batch, history_len, feature_num] - 1=已观测, 0=缺失
        
        返回:
            encoded: [batch, history_len, d_model]
        """
        # 构造输入
        X = history
        masks = missing_mask if missing_mask is not None else torch.ones_like(history)
        
        # ============================================
        # 关键: 将 [X, masks] 拼接作为编码器输入
        # 这样 Self-Attention 能区分真实观测值和填充的 0
        # ============================================
        
        # 第一个 DMSA 块
        if self.saits_encoder.input_with_mask:
            # 拼接: [batch, history_len, feature_num*2]
            input_X_for_first = torch.cat([X, masks], dim=2)
            # Self-Attention 会学习:
            # - 真实观测位置 (mask=1) 的模式
            # - 忽略或降权缺失位置 (mask=0)
        else:
            input_X_for_first = X
        
        input_X_for_first = self.saits_encoder.embedding_1(input_X_for_first)
        enc_output = self.saits_encoder.dropout(
            self.saits_encoder.position_enc(input_X_for_first)
        )
        
        if self.saits_encoder.param_sharing_strategy == "between_group":
            for _ in range(self.saits_encoder.n_groups):
                for encoder_layer in self.saits_encoder.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.saits_encoder.layer_stack_for_first_block:
                for _ in range(self.saits_encoder.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)
        
        # 返回编码表示: [batch, history_len, d_model]
        return enc_output
    
    def forward(self, history, future=None, missing_mask=None, stage='train'):
        """
        前向传播
        
        参数:
            history: [batch, history_len, feature_num] 历史数据
            future: [batch, forecast_horizon, feature_num] 未来数据 (仅训练时需要)
            missing_mask: [batch, history_len, feature_num] 缺失掩码 (可选)
            stage: 'train', 'val', 'test'
        
        返回:
            results: 字典,包含 forecast, loss 等信息
        """
        # 1. 编码历史
        encoded_history = self.encode_history(history, missing_mask)
        
        # 2. 预测未来
        forecast = self.forecast_head(encoded_history)
        
        # 3. 计算损失 (仅在训练/验证时)
        if stage in ['train', 'val'] and future is not None:
            # 预测损失
            forecast_loss_mse = self.mse_loss(forecast, future)
            forecast_loss_mae = self.mae_loss(forecast, future)
            
            # 可选: 历史重建损失 (继承 SAITS 的双任务思想)
            # 这里简化处理,只用预测损失
            total_loss = forecast_loss_mse
            
            return {
                'forecast': forecast,
                'total_loss': total_loss,
                'forecast_loss_mse': forecast_loss_mse,
                'forecast_loss_mae': forecast_loss_mae,
            }
        else:
            # 测试阶段只返回预测
            return {
                'forecast': forecast
            }


class SAITS_Forecaster_V2(nn.Module):
    """
    SAITS Forecaster 增强版
    
    与 V1 的区别:
    1. 添加历史重建任务 (双任务联合优化,继承 SAITS 思想)
    2. 支持更复杂的解码器 (可扩展为 Seq2Seq)
    
    训练策略:
        Loss = α * Reconstruction_Loss + β * Forecast_Loss
    """
    def __init__(self, *args, reconstruction_weight=0.3, forecast_weight=1.0, **kwargs):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.forecast_weight = forecast_weight
        
        # 初始化基础模型
        self.base_model = SAITS_Forecaster(*args, **kwargs)
        
        # 新增: 重建投影层 (用于历史重建任务)
        # 从编码器的输出维度 d_model 投影到特征维度 d_feature
        d_model = kwargs.get('d_model', 256)  # 从参数中获取 d_model
        self.reconstruction_proj = nn.Linear(d_model, self.base_model.d_feature)
    
    def forward(self, history, future=None, missing_mask=None, stage='train'):
        """前向传播 (带重建任务)"""
        # 1. 编码历史
        encoded_history = self.base_model.encode_history(history, missing_mask)
        
        # 2. 任务A: 重建历史 (类似 SAITS 的 ORT)
        reconstructed_history = self.reconstruction_proj(encoded_history)
        
        # 3. 任务B: 预测未来
        forecast = self.base_model.forecast_head(encoded_history)
        
        # 4. 计算损失
        if stage in ['train', 'val'] and future is not None:
            # 重建损失 (增强编码器的表示能力)
            recon_loss = self.base_model.mae_loss(reconstructed_history, history)
            
            # 预测损失
            forecast_loss_mse = self.base_model.mse_loss(forecast, future)
            forecast_loss_mae = self.base_model.mae_loss(forecast, future)
            
            # 联合优化 (继承 SAITS 的双任务思想!)
            total_loss = (
                self.reconstruction_weight * recon_loss +
                self.forecast_weight * forecast_loss_mse
            )
            
            return {
                'forecast': forecast,
                'reconstructed_history': reconstructed_history,
                'total_loss': total_loss,
                'reconstruction_loss': recon_loss,
                'forecast_loss_mse': forecast_loss_mse,
                'forecast_loss_mae': forecast_loss_mae,
            }
        else:
            return {
                'forecast': forecast,
                'reconstructed_history': reconstructed_history
            }


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 SAITS_Forecaster")
    print("=" * 60)
    
    # 模型参数
    batch_size = 8
    history_len = 36
    forecast_horizon = 12
    feature_num = 37
    d_model = 256
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SAITS_Forecaster(
        n_groups=2,
        n_group_inner_layers=2,
        history_len=history_len,
        forecast_horizon=forecast_horizon,
        d_feature=feature_num,
        d_model=d_model,
        d_inner=128,
        n_head=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        device=device,
        input_with_mask=True,
        diagonal_attention_mask=True,
        param_sharing_strategy='inner_group'
    ).to(device)
    
    # 测试数据
    history = torch.randn(batch_size, history_len, feature_num).to(device)
    future = torch.randn(batch_size, forecast_horizon, feature_num).to(device)
    missing_mask = torch.ones(batch_size, history_len, feature_num).to(device)
    
    # 前向传播
    results = model(history, future, missing_mask, stage='train')
    
    print(f"\n✅ 模型测试成功!")
    print(f"   输入 (历史): {history.shape}")
    print(f"   输出 (预测): {results['forecast'].shape}")
    print(f"   总损失: {results['total_loss'].item():.4f}")
    print(f"   预测 MAE: {results['forecast_loss_mae'].item():.4f}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   可训练参数: {total_params:,}")
