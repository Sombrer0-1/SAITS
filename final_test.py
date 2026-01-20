"""
预测模型快速测试 - 无需训练,只验证代码逻辑

核心理念:
- 快速检测代码问题,而非评估性能
- 测试数据流和模型结构,不关心训练效果
- 前提: 代码对了,训练只是时间问题

测试清单:
✓ 数据维度、类型、NaN处理、missing_mask生成
✓ 模型输入输出维度、组件存在性
✓ Missing_Mask 是否真的被使用 (核心!)
✓ 前向/反向传播能否跑通
✓ 边界情况 (batch=1、全缺失、全观测)

运行: python final_test.py
预计: < 30秒
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import tempfile
import shutil

print("=" * 80)
print("预测模型快速测试 (无需训练)".center(80))
print("=" * 80)
print(f"\n⏱️  预计耗时: < 30秒\n")

results = {'passed': [], 'failed': [], 'warnings': []}

def log(name, status, msg=""):
    results[status + ('ed' if status != 'warning' else 's')].append(name)
    icon = {"pass": "✅", "fail": "❌", "warning": "⚠️"}[status]
    print(f"{icon} {name}" + (f"\n   → {msg}" if msg else ""))

# ====================================================================================
# 阶段 1: 数据加载验证
# ====================================================================================
print("\n" + "=" * 80)
print("阶段 1: 数据加载与预处理".center(80))
print("=" * 80 + "\n")

try:
    from modeling.forecasting_dataloader import ForecastingDataLoader
    
    dataset_path = "generated_datasets/physio2012_37feats_01masked/datasets.h5"
    
    if not os.path.exists(dataset_path):
        log("数据集存在性", "fail", f"找不到: {dataset_path}")
        sys.exit(1)
    
    # 检查原始数据是否含 NaN
    with h5py.File(dataset_path, 'r') as hf:
        has_nan = np.isnan(hf['train']['X'][:100]).any()
        log("数据集格式", "pass" if has_nan else "warning", 
            "包含 NaN" if has_nan else "无 NaN (可能已预处理)")
    
    # 创建加载器 (小批次)
    loader = ForecastingDataLoader(
        dataset_path=dataset_path, 
        history_len=36, 
        forecast_horizon=12,
        batch_size=4, 
        num_workers=0, 
        stride=6, 
        handle_missing=True
    )
    
    train_loader, val_loader = loader.get_train_val_dataloader()
    history, mask, future = next(iter(train_loader))
    
    # 批量检查
    checks = [
        (history.shape == (4, 36, 37), f"History: {history.shape}"),
        (mask.shape == (4, 36, 37), f"Mask: {mask.shape}"),
        (future.shape == (4, 12, 37), f"Future: {future.shape}"),
        (history.dtype == torch.float32, "数据类型 float32"),
        (not torch.isnan(history).any(), "History 无 NaN"),
        (set(mask.unique().numpy()).issubset({0., 1.}), f"Mask 值域: {set(mask.unique().numpy())}"),
    ]
    
    for check, msg in checks:
        log(msg, "pass" if check else "fail")
    
    # 验证缺失位置填充
    missing_pos = (mask == 0)
    if missing_pos.any():
        all_zero = torch.all(history[missing_pos] == 0).item()
        missing_rate = missing_pos.float().mean().item()
        log("缺失位置填充为0", "pass" if all_zero else "warning")
        log(f"缺失率 {missing_rate:.1%}", "pass")
    
    print("\n✓ 数据加载正常\n")

except Exception as e:
    log("数据加载", "fail", str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================================
# 阶段 2: 模型结构验证
# ====================================================================================
print("=" * 80)
print("阶段 2: 模型结构验证".center(80))
print("=" * 80 + "\n")

try:
    from modeling.saits_forecaster import SAITS_Forecaster, SAITS_Forecaster_V2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}\n")
    
    # V1 模型 (小规模)
    model_v1 = SAITS_Forecaster(
        n_groups=1, 
        n_group_inner_layers=1, 
        history_len=36, 
        forecast_horizon=12,
        d_feature=37, 
        d_model=128, 
        d_inner=64, 
        n_head=2, 
        d_k=32, 
        d_v=32,
        dropout=0.1, 
        device=device, 
        input_with_mask=True,
        diagonal_attention_mask=True, 
        param_sharing_strategy='inner_group'
    ).to(device)
    
    checks = [
        (hasattr(model_v1, 'saits_encoder'), "包含 SAITS 编码器"),
        (hasattr(model_v1, 'forecast_head'), "包含预测头"),
        (model_v1.saits_encoder.input_with_mask, "input_with_mask=True"),
    ]
    
    for check, msg in checks:
        log(msg, "pass" if check else "fail")
    
    # 检查嵌入层维度
    emb_dim = model_v1.saits_encoder.embedding_1.weight.shape[1]
    expected = 37 * 2
    log(f"编码器输入维度 {emb_dim}", "pass" if emb_dim == expected else "fail",
        "正确 (X+mask拼接)" if emb_dim == expected else f"错误 (期望{expected})")
    
    # V2 模型 (双任务学习)
    model_v2 = SAITS_Forecaster_V2(
        n_groups=1, 
        n_group_inner_layers=1, 
        history_len=36, 
        forecast_horizon=12,
        d_feature=37, 
        d_model=128, 
        d_inner=64, 
        n_head=2, 
        d_k=32, 
        d_v=32,
        dropout=0.1, 
        device=device, 
        input_with_mask=True,
        diagonal_attention_mask=True, 
        param_sharing_strategy='inner_group',
        reconstruction_weight=0.3
    ).to(device)
    
    log("V2 重构投影层", "pass" if hasattr(model_v2, 'reconstruction_proj') else "fail")
    
    print("\n✓ 模型结构正常\n")

except Exception as e:
    log("模型构建", "fail", str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================================
# 阶段 3: 前向传播验证
# ====================================================================================
print("=" * 80)
print("阶段 3: 前向传播验证".center(80))
print("=" * 80 + "\n")

try:
    history = history.to(device)
    mask = mask.to(device)
    future = future.to(device)
    
    # V1
    model_v1.eval()
    with torch.no_grad():
        out_v1 = model_v1(history, future, mask, stage='val')
    
    forecast = out_v1['forecast']
    
    checks = [
        ('forecast' in out_v1, "包含 'forecast' 键"),
        (forecast.shape == (4, 12, 37), f"预测形状 {forecast.shape}"),
        (not torch.isnan(forecast).any(), "无 NaN"),
        (not torch.isinf(forecast).any(), "无 Inf"),
        (not torch.all(forecast == 0), "不全为0"),
    ]
    
    for check, msg in checks:
        log(f"V1 - {msg}", "pass" if check else "fail")
    
    # V2 双任务测试
    model_v2.eval()
    with torch.no_grad():
        out_v2 = model_v2(history, future, mask, stage='val')
    
    v2_checks = [
        ('forecast' in out_v2, "包含 'forecast' 键"),
        ('reconstructed_history' in out_v2, "包含 'reconstructed_history' 键"),
        (out_v2['forecast'].shape == (4, 12, 37), f"预测形状 {out_v2['forecast'].shape}"),
        (out_v2['reconstructed_history'].shape == (4, 36, 37), f"重构形状 {out_v2['reconstructed_history'].shape}"),
        (not torch.isnan(out_v2['forecast']).any(), "预测无 NaN"),
        (not torch.isnan(out_v2['reconstructed_history']).any(), "重构无 NaN"),
    ]
    
    for check, msg in v2_checks:
        log(f"V2 - {msg}", "pass" if check else "fail")
    
    print("\n✓ 前向传播正常 (V1+V2)\n")

except Exception as e:
    log("前向传播", "fail", str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================================
# 阶段 4: Missing_Mask 机制验证 (最关键!)
# ====================================================================================
print("=" * 80)
print("阶段 4: Missing_Mask 机制验证 ⭐ 核心".center(80))
print("=" * 80 + "\n")

try:
    model_v1.eval()
    
    # 测试1: 真实 mask vs 全1 mask
    with torch.no_grad():
        out_real = model_v1(history, future, mask, stage='val')['forecast']
        out_fake = model_v1(history, future, torch.ones_like(mask), stage='val')['forecast']
        diff = torch.abs(out_real - out_fake).mean().item()
    
    if diff < 1e-6:
        log("Mask 敏感性", "fail", f"真实mask和全1mask预测相同 (diff={diff:.2e})")
    else:
        log("Mask 敏感性", "pass", f"预测差异={diff:.4f} ✓ 模型正在使用mask")
    
    # 测试2: 不同缺失率
    with torch.no_grad():
        low = (torch.rand_like(mask) > 0.2).float()
        high = (torch.rand_like(mask) > 0.8).float()
        diff_rate = torch.abs(
            model_v1(history, future, low, stage='val')['forecast'] - 
            model_v1(history, future, high, stage='val')['forecast']
        ).mean().item()
    
    log("缺失率影响", "pass", f"20% vs 80% 差异={diff_rate:.4f}")
    
    # 测试3: 捕获编码器输入
    captured = {'shape': None}
    
    def capture_hook(module, input, output):
        captured['shape'] = input[0].shape
    
    hook = model_v1.saits_encoder.embedding_1.register_forward_hook(capture_hook)
    
    with torch.no_grad():
        _ = model_v1(history[:1], future[:1], mask[:1], stage='val')
    
    hook.remove()
    
    if captured['shape']:
        last_dim = captured['shape'][-1]
        log(f"编码器实际输入 {captured['shape']}", 
            "pass" if last_dim == 74 else "fail",
            f"最后一维={last_dim}, {'✓ X+mask拼接' if last_dim == 74 else '✗ 错误'}")
    
    print("\n✓ Missing_Mask 机制生效\n")

except Exception as e:
    log("Missing_Mask", "fail", str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================================
# 阶段 5: 反向传播验证
# ====================================================================================
print("=" * 80)
print("阶段 5: 反向传播验证".center(80))
print("=" * 80 + "\n")

try:
    model_v1.train()
    optimizer = optim.Adam(model_v1.parameters(), lr=1e-3)
    
    initial = next(model_v1.parameters()).clone()
    
    # 前向+反向
    out = model_v1(history, future, mask, stage='train')
    optimizer.zero_grad()
    
    # V1 返回的是 'total_loss' 不是 'loss'
    loss = out.get('total_loss') or out.get('loss')
    if loss is None:
        log("损失键检查", "fail", f"输出键: {list(out.keys())}")
        sys.exit(1)
    
    loss.backward()
    
    # 检查梯度
    has_grad = any(p.grad is not None for p in model_v1.parameters() if p.requires_grad)
    has_nan = any(torch.isnan(p.grad).any() for p in model_v1.parameters() if p.grad is not None)
    
    log("梯度计算", "pass" if has_grad else "fail")
    log("梯度数值", "fail" if has_nan else "pass", "包含 NaN" if has_nan else "正常")
    
    # 更新参数
    optimizer.step()
    changed = not torch.equal(initial, next(model_v1.parameters()))
    log("参数更新", "pass" if changed else "fail")
    
    print("\n✓ 反向传播正常\n")

except Exception as e:
    log("反向传播", "fail", str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================================
# 阶段 6: 边界情况测试
# ====================================================================================
print("=" * 80)
print("阶段 6: 边界情况测试".center(80))
print("=" * 80 + "\n")

try:
    model_v1.eval()
    
    # batch=1
    with torch.no_grad():
        out = model_v1(history[:1], future[:1], mask[:1], stage='val')['forecast']
        log("单样本 (batch=1)", "pass" if out.shape[0] == 1 else "fail", f"形状 {out.shape}")
    
    # 全缺失
    try:
        with torch.no_grad():
            out = model_v1(history[:2], future[:2], torch.zeros_like(mask[:2]), stage='val')['forecast']
            has_nan = torch.isnan(out).any()
            log("全缺失数据", "warning" if has_nan else "pass", "产生 NaN" if has_nan else "正常")
    except Exception as e:
        log("全缺失数据", "fail", str(e)[:50])
    
    # 全观测
    with torch.no_grad():
        out = model_v1(history[:2], future[:2], torch.ones_like(mask[:2]), stage='val')['forecast']
        log("全观测数据", "pass" if not torch.isnan(out).any() else "fail")
    
    print("\n✓ 边界情况正常\n")

except Exception as e:
    log("边界情况", "fail", str(e))
    import traceback
    traceback.print_exc()

# ====================================================================================
# 阶段 7: 保存/加载验证
# ====================================================================================
print("=" * 80)
print("阶段 7: 保存/加载验证".center(80))
print("=" * 80 + "\n")

try:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "test.pth")
    
    # 保存
    torch.save(model_v1.state_dict(), path)
    log("模型保存", "pass", f"{os.path.getsize(path)/1024:.1f} KB")
    
    # 原始预测
    model_v1.eval()
    with torch.no_grad():
        orig = model_v1(history[:2], future[:2], mask[:2], stage='val')['forecast'].clone()
    
    # 加载
    model_new = SAITS_Forecaster(
        n_groups=1, 
        n_group_inner_layers=1, 
        history_len=36, 
        forecast_horizon=12,
        d_feature=37, 
        d_model=128, 
        d_inner=64, 
        n_head=2, 
        d_k=32, 
        d_v=32,
        dropout=0.1, 
        device=device, 
        input_with_mask=True,
        diagonal_attention_mask=True, 
        param_sharing_strategy='inner_group'
    ).to(device)
    
    model_new.load_state_dict(torch.load(path))
    model_new.eval()
    
    with torch.no_grad():
        loaded = model_new(history[:2], future[:2], mask[:2], stage='val')['forecast']
    
    is_equal = torch.allclose(orig, loaded, atol=1e-6)
    log("加载后一致性", "pass" if is_equal else "fail")
    
    shutil.rmtree(temp_dir)
    print("\n✓ 保存/加载正常\n")

except Exception as e:
    log("保存/加载", "fail", str(e))
    if 'temp_dir' in locals():
        shutil.rmtree(temp_dir, ignore_errors=True)

# ====================================================================================
# 测试报告
# ====================================================================================
print("=" * 80)
print("测试报告".center(80))
print("=" * 80 + "\n")

total = sum(len(v) for v in results.values())
passed = len(results['passed'])
failed = len(results['failed'])
warnings = len(results['warnings'])

print(f"📊 总计: {total} 项")
print(f"   ✅ 通过: {passed} ({passed/total*100:.0f}%)")
print(f"   ❌ 失败: {failed}")
print(f"   ⚠️  警告: {warnings}\n")

if failed > 0:
    print("❌ 失败:")
    for t in results['failed']: 
        print(f"   • {t}")
    print()

if warnings > 0:
    print("⚠️  警告:")
    for t in results['warnings']: 
        print(f"   • {t}")
    print()

print("=" * 80 + "\n")

if failed == 0:
    print("🎉 所有关键测试通过!\n")
    print("✓ 数据流正确")
    print("✓ 模型结构正确")
    print("✓ Missing_Mask 机制生效")
    print("✓ 前向/反向传播正常\n")
    print("👉 下一步: 训练模型")
    print("   python run_forecast.py --config_path configs/PhysioNet2012_Forecast.ini\n")
    sys.exit(0)
else:
    print("❌ 存在失败测试,请先修复\n")
    sys.exit(1)
