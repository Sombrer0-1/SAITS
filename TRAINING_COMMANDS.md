# SAITS 模型训练与测试指令手册

本文档记录了 SAITS 项目中不同数据集的训练和测试命令。

## 📋 目录
- [PhysioNet-2012 数据集](#1-physionet-2012-数据集)
- [AirQuality 数据集](#2-airquality-数据集)
- [Electricity 数据集](#3-electricity-数据集)
- [NRTSI AirQuality 数据集](#4-nrtsi-airquality-数据集)

---

## 1. PhysioNet-2012 数据集

### 📊 数据集信息
- **名称**: PhysioNet Challenge 2012 (医疗 ICU 数据)
- **特征数**: 37 个生理指标
- **时间步长**: 48 小时
- **样本数**: 训练集 7,672 条记录

### 🚀 训练指令
```powershell
# 使用最佳配置训练 PhysioNet2012 数据集
python run_models.py --config_path configs/PhysioNet2012_SAITS_best.ini
```

### 🧪 测试指令
```powershell
# 测试已训练的 PhysioNet2012 模型
python run_models.py --config_path configs/PhysioNet2012_SAITS_best.ini --test_mode
```

### 📝 注意事项
- 训练前请确保已生成数据集,运行 `dataset_generating_scripts/gene_PhysioNet2012_dataset.py`
- 测试前需在配置文件中指定已训练模型的路径
- 预期训练时间: 约 1 小时 (GPU: RTX 3090)
- 预期性能: MAE ≈ 0.19, RMSE ≈ 0.42

---

## 2. AirQuality 数据集

### 📊 数据集信息
- **名称**: UCI Beijing Air Quality (北京空气质量数据)
- **特征数**: 多个空气质量指标 (PM2.5, PM10, SO2, NO2, CO, O3 等)
- **时间步长**: 24 小时
- **缺失率**: 10% (人工掩码)

### 🚀 训练指令
```powershell
# 使用最佳配置训练 AirQuality 数据集
python run_models.py --config_path configs/AirQuality_SAITS_best.ini
```

### 🧪 测试指令
```powershell
# 测试已训练的 AirQuality 模型
python run_models.py --config_path configs/AirQuality_SAITS_best.ini --test_mode
```

### 📝 注意事项
- 训练前请确保已生成数据集,运行 `dataset_generating_scripts/gene_UCI_BeijingAirQuality_dataset.py`
- 数据集序列长度: 24 (seqlen24)
- 人工缺失掩码比例: 10% (01masked)

---

## 3. Electricity 数据集

### 📊 数据集信息
- **名称**: UCI Electricity Load Diagrams (电力负荷数据)
- **特征数**: 321 个客户的电力消耗数据
- **时间步长**: 100 小时
- **缺失率**: 10% (人工掩码)

### 🚀 训练指令
```powershell
# 使用最佳配置训练 Electricity 数据集
python run_models.py --config_path configs/Electricity_SAITS_best.ini
```

### 🧪 测试指令
```powershell
# 测试已训练的 Electricity 模型
python run_models.py --config_path configs/Electricity_SAITS_best.ini --test_mode
```

### 📝 注意事项
- 训练前请确保已生成数据集,运行 `dataset_generating_scripts/gene_UCI_electricity_dataset.py`
- 数据集序列长度: 100 (seqlen100)
- 人工缺失掩码比例: 10% (01masked)

---

## 4. NRTSI AirQuality 数据集

### 📊 数据集信息
- **名称**: NRTSI (Non-Recurrent Time Series Imputation) AirQuality
- **特征数**: 空气质量指标
- **说明**: 用于与 NRTSI 论文进行对比实验的数据集配置

### 🚀 训练指令
```powershell
# 使用 NRTSI 对比配置训练 AirQuality 数据集
python run_models.py --config_path configs/NRTSI_AirQuality_SAITS_best.ini
```

### 🧪 测试指令
```powershell
# 测试 NRTSI AirQuality 模型
python run_models.py --config_path configs/NRTSI_AirQuality_SAITS_best.ini --test_mode
```

### 📝 注意事项
- 此配置用于与 NRTSI 论文进行公平对比
- 训练前请确保已生成对应的 NRTSI 数据集,运行 `dataset_generating_scripts/gene_NRTSI_dataset.py`

---

## 🔧 通用说明

### 前置准备
1. **创建结果保存目录**:
   ```powershell
   mkdir NIPS_results
   ```

2. **后台运行训练** (可选):
   ```powershell
   # Windows PowerShell 后台运行示例
   Start-Process python -ArgumentList "run_models.py --config_path configs/PhysioNet2012_SAITS_best.ini" -NoNewWindow -RedirectStandardOutput "NIPS_results/PhysioNet2012_SAITS_best.out"
   ```

3. **查看训练日志**:
   ```powershell
   Get-Content NIPS_results/PhysioNet2012_SAITS_best.out -Wait -Tail 50
   ```

### 配置文件修改
测试前需要修改配置文件中的以下参数:
- `[test]` 部分的 `model_saving_dir`: 指向训练好的模型时间戳目录 (例如: `2026-01-16_T20_18_27`)
- `[test]` 部分的 `step`: 指向最佳模型的训练步数 (例如: `159`)

### 警告信息处理
如果遇到 sklearn 的 `pkg_resources` 警告信息,已在 `run_models.py` 中添加了警告过滤器:
```python
# 已在代码第 27-29 行添加
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
```

### 训练监控
训练过程中可以查看以下信息:
- 训练日志: `NIPS_results/{model_name}/logs_{timestamp}/train.log`
- TensorBoard 可视化 (如果启用): `NIPS_results/{model_name}/tensorboard_{timestamp}`
- 模型检查点: `NIPS_results/{model_name}/models/{timestamp}/`
- 测试结果: `NIPS_results/{model_name}/step_{best_step}/imputations.h5`

---

## 📊 性能参考指标

根据论文和实际测试,各数据集的预期性能如下:

| 数据集 | MAE | RMSE | MRE |
|--------|-----|------|-----|
| PhysioNet-2012 | ~0.19 | ~0.42 | ~0.28 |
| AirQuality | 待测试 | 待测试 | 待测试 |
| Electricity | 待测试 | 待测试 | 待测试 |

---

## 🐛 常见问题

### 1. 数据集未找到
**错误**: `FileNotFoundError: Dataset not found`
**解决**: 运行对应的数据集生成脚本,例如:
```powershell
python dataset_generating_scripts/gene_PhysioNet2012_dataset.py
```

### 2. 模型路径错误
**错误**: `FileNotFoundError: Model checkpoint not found`
**解决**: 检查配置文件 `[test]` 部分的 `model_saving_dir` 和 `step` 参数是否正确

### 3. CUDA 内存不足
**错误**: `RuntimeError: CUDA out of memory`
**解决**: 减小配置文件中的 `batch_size` 参数,或使用 CPU 训练 (修改 `device` 参数为 `cpu`)

### 4. 编码错误
**错误**: `UnicodeDecodeError: 'gbk' codec can't decode`
**解决**: 已在 `run_models.py` 第 524 行添加 `encoding='utf-8'` 参数,确保使用最新版本代码

---

## 📚 相关资源

- **论文**: [SAITS: Self-Attention-based Imputation for Time Series](https://arxiv.org/abs/2202.08516)
- **PyPOTS 工具库**: [https://github.com/WenjieDu/PyPOTS](https://github.com/WenjieDu/PyPOTS)
- **数据集说明**: 见 `dataset_generating_scripts/README.md`
- **超参数搜索**: 见 `NNI_tuning/` 目录

---

**最后更新**: 2026年1月17日
**维护者**: Sombrer0-1
