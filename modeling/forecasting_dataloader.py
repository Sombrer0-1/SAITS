"""
数据加载器 - 用于时间序列预测任务

与 unified_dataloader.py 的区别:
- unified_dataloader.py: 用于插补任务,返回 (X, missing_mask, X_holdout)
- forecasting_dataloader.py: 用于预测任务,返回 (history, future)

Created for SAITS Forecasting Extension
License: MIT
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ForecastingDataset(Dataset):
    """
    时间序列预测数据集
    
    将长时间序列切分为 (历史窗口, 未来窗口) 对
    
    参数:
        X: 完整时间序列数据 [n_samples, seq_len, feature_num]
        history_len: 历史窗口长度
        forecast_horizon: 预测未来的步数
        stride: 滑动窗口步长
    """
    def __init__(self, X, history_len, forecast_horizon, stride=1):
        super().__init__()
        self.X = X
        self.history_len = history_len
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        # 计算有效样本数
        self.n_samples = X.shape[0]
        self.seq_len = X.shape[1]
        self.feature_num = X.shape[2]
        
        # 生成所有有效的 (history, future) 窗口索引
        self.valid_indices = []
        for sample_idx in range(self.n_samples):
            for start_idx in range(0, self.seq_len - history_len - forecast_horizon + 1, stride):
                self.valid_indices.append((sample_idx, start_idx))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        sample_idx, start_idx = self.valid_indices[idx]
        
        # 提取历史窗口
        history = self.X[sample_idx, start_idx:start_idx + self.history_len, :]
        
        # 提取未来窗口
        future_start = start_idx + self.history_len
        future = self.X[sample_idx, future_start:future_start + self.forecast_horizon, :]
        
        return (
            torch.from_numpy(history.astype('float32')),
            torch.from_numpy(future.astype('float32'))
        )


class ForecastingDatasetWithMissing(Dataset):
    """
    支持缺失数据的预测数据集
    
    数据处理流程:
    1. 从 HDF5 读取原始 X (包含 NaN 标记的缺失值)
    2. 生成 missing_mask: ~np.isnan(X) → (1=已观测, 0=缺失)
    3. 填充 NaN 为 0: np.nan_to_num(X, nan=0.0)
    4. 将 [X_filled, missing_mask] 一起传给模型
    
    返回: (history, missing_mask, future)
    
    关键设计:
    - missing_mask 告诉模型哪些位置是真实观测 (1),哪些是填充值 (0)
    - 模型通过 input_with_mask=True 利用 mask 信息
    - Self-Attention 会根据 mask 调整注意力权重
    """
    def __init__(self, X, missing_mask, history_len, forecast_horizon, stride=1):
        super().__init__()
        self.X = X
        self.missing_mask = missing_mask  # 新增: 存储真实的 missing_mask
        self.history_len = history_len
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        
        self.n_samples = X.shape[0]
        self.seq_len = X.shape[1]
        self.feature_num = X.shape[2]
        
        # 生成有效索引
        self.valid_indices = []
        for sample_idx in range(self.n_samples):
            for start_idx in range(0, self.seq_len - history_len - forecast_horizon + 1, stride):
                self.valid_indices.append((sample_idx, start_idx))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        sample_idx, start_idx = self.valid_indices[idx]
        
        # 提取历史窗口 (NaN 已填充为 0)
        history = self.X[sample_idx, start_idx:start_idx + self.history_len, :]
        
        # 提取对应的 missing_mask (从 NaN 生成,1=观测,0=缺失)
        history_mask = self.missing_mask[sample_idx, start_idx:start_idx + self.history_len, :]
        
        # 提取未来窗口
        future_start = start_idx + self.history_len
        future = self.X[sample_idx, future_start:future_start + self.forecast_horizon, :]
        
        return (
            torch.from_numpy(history.astype('float32')),
            torch.from_numpy(history_mask.astype('float32')),
            torch.from_numpy(future.astype('float32'))
        )


class ForecastingDataLoader:
    """
    预测任务的统一数据加载器
    
    功能:
    1. 从 H5 文件加载数据
    2. 按照 (历史, 未来) 窗口划分
    3. 创建 train/val/test DataLoader
    """
    def __init__(
        self, 
        dataset_path,
        history_len,
        forecast_horizon,
        batch_size,
        num_workers=4,
        stride=1,
        handle_missing=False
    ):
        """
        参数:
            dataset_path: H5 数据集路径
            history_len: 历史窗口长度
            forecast_horizon: 预测未来的步数
            batch_size: 批大小
            num_workers: DataLoader 工作进程数
            stride: 滑动窗口步长
            handle_missing: 是否处理缺失值
        """
        self.dataset_path = dataset_path
        self.history_len = history_len
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.handle_missing = handle_missing
        
        # 加载数据
        self._load_data()
        
        # 创建数据集
        self._create_datasets()
    
    def _load_data(self):
        """从 H5 文件加载数据,并从 NaN 生成 missing_mask"""
        with h5py.File(self.dataset_path, 'r') as hf:
            # 加载原始数据 X (包含 NaN)
            train_X_raw = hf['train']['X'][:]
            val_X_raw = hf['val']['X'][:]
            test_X_raw = hf['test']['X'][:]
            
            # 从 NaN 生成 missing_mask (1=已观测, 0=缺失)
            self.train_missing_mask = (~np.isnan(train_X_raw)).astype('float32')
            self.val_missing_mask = (~np.isnan(val_X_raw)).astype('float32')
            self.test_missing_mask = (~np.isnan(test_X_raw)).astype('float32')
            
            # 填充 NaN 为 0 (供模型使用)
            self.train_X = np.nan_to_num(train_X_raw, nan=0.0)
            self.val_X = np.nan_to_num(val_X_raw, nan=0.0)
            self.test_X = np.nan_to_num(test_X_raw, nan=0.0)
            
            # 记录数据维度
            self.feature_num = self.train_X.shape[2]
            self.original_seq_len = self.train_X.shape[1]
            
            # 统计缺失率
            train_missing_rate = 1 - self.train_missing_mask.mean()
            val_missing_rate = 1 - self.val_missing_mask.mean()
            test_missing_rate = 1 - self.test_missing_mask.mean()
        
        print(f"✅ 数据加载完成:")
        print(f"   训练集: {self.train_X.shape}, 缺失率: {train_missing_rate:.2%}")
        print(f"   验证集: {self.val_X.shape}, 缺失率: {val_missing_rate:.2%}")
        print(f"   测试集: {self.test_X.shape}, 缺失率: {test_missing_rate:.2%}")
        print(f"   特征维度: {self.feature_num}")
        print(f"   历史长度: {self.history_len}")
        print(f"   预测步数: {self.forecast_horizon}")
    
    def _create_datasets(self):
        """创建 PyTorch Dataset"""
        if self.handle_missing:
            # 使用带 missing_mask 的数据集
            self.train_set = ForecastingDatasetWithMissing(
                self.train_X, self.train_missing_mask,
                self.history_len, self.forecast_horizon, self.stride
            )
            self.val_set = ForecastingDatasetWithMissing(
                self.val_X, self.val_missing_mask,
                self.history_len, self.forecast_horizon, self.stride
            )
            self.test_set = ForecastingDatasetWithMissing(
                self.test_X, self.test_missing_mask,
                self.history_len, self.forecast_horizon, self.stride
            )
        else:
            # 不考虑缺失值
            self.train_set = ForecastingDataset(
                self.train_X, self.history_len, self.forecast_horizon, self.stride
            )
            self.val_set = ForecastingDataset(
                self.val_X, self.history_len, self.forecast_horizon, self.stride
            )
            self.test_set = ForecastingDataset(
                self.test_X, self.history_len, self.forecast_horizon, self.stride
            )
        
        print(f"✅ 数据集创建完成:")
        print(f"   训练样本: {len(self.train_set)}")
        print(f"   验证样本: {len(self.val_set)}")
        print(f"   测试样本: {len(self.test_set)}")
    
    def get_train_val_dataloader(self):
        """获取训练和验证 DataLoader"""
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False
        )
        
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def get_test_dataloader(self):
        """获取测试 DataLoader"""
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False
        )
        
        return test_loader


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 ForecastingDataLoader")
    print("=" * 60)
    
    # 示例: 加载 PhysioNet2012 数据集
    dataset_path = "generated_datasets/physio2012_37feats_01masked/datasets.h5"
    
    if os.path.exists(dataset_path):
        loader = ForecastingDataLoader(
            dataset_path=dataset_path,
            history_len=36,      # 使用前36步作为历史
            forecast_horizon=12, # 预测未来12步
            batch_size=64,
            stride=6,            # 滑动窗口步长
            handle_missing=True  # 处理缺失值
        )
        
        train_loader, val_loader = loader.get_train_val_dataloader()
        
        # 测试一个批次
        for batch_data in train_loader:
            if len(batch_data) == 3:
                history, mask, future = batch_data
                print(f"\n✅ 批次数据形状 (带缺失处理):")
                print(f"   历史: {history.shape}")
                print(f"   掩码: {mask.shape}")
                print(f"   未来: {future.shape}")
            else:
                history, future = batch_data
                print(f"\n✅ 批次数据形状:")
                print(f"   历史: {history.shape}")
                print(f"   未来: {future.shape}")
            break
    else:
        print(f"❌ 数据集文件不存在: {dataset_path}")
        print("   请先运行数据生成脚本")
