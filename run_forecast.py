"""
时间序列预测任务的训练脚本

与 run_models.py 的区别:
- run_models.py: 插补任务 (Imputation)
- run_forecast.py: 预测任务 (Forecasting)

设计原则:
1. 最大化复用 run_models.py 的工具函数
2. 保持代码风格一致
3. 复用 SAITS 的编码器架构

使用方法:
    # 训练
    python run_forecast.py --config_path configs/PhysioNet2012_Forecast.ini
    
    # 测试
    python run_forecast.py --config_path configs/PhysioNet2012_Forecast.ini --test_mode

Created for SAITS Forecasting Extension
License: MIT
"""

import argparse
import math
import os
import warnings
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 关闭警告
warnings.filterwarnings("ignore")

try:
    import nni
except ImportError:
    pass

from Global_Config import RANDOM_SEED
from modeling.saits_forecaster import SAITS_Forecaster, SAITS_Forecaster_V2
from modeling.forecasting_dataloader import ForecastingDataLoader
from modeling.utils import (
    Controller,
    setup_logger,
    save_model,
    load_model,
)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 支持的预测模型字典
FORECAST_MODEL_DICT = {
    "SAITS_Forecaster": SAITS_Forecaster,
    "SAITS_Forecaster_V2": SAITS_Forecaster_V2,
}

OPTIMIZER = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}


def read_forecast_arguments(arg_parser, cfg_parser):
    """读取预测任务的配置参数"""
    # 文件路径
    arg_parser.dataset_base_dir = cfg_parser.get("file_path", "dataset_base_dir")
    arg_parser.result_saving_base_dir = cfg_parser.get("file_path", "result_saving_base_dir")
    
    # 数据集信息
    arg_parser.history_len = cfg_parser.getint("dataset", "history_len")
    arg_parser.forecast_horizon = cfg_parser.getint("dataset", "forecast_horizon")
    arg_parser.batch_size = cfg_parser.getint("dataset", "batch_size")
    arg_parser.num_workers = cfg_parser.getint("dataset", "num_workers")
    arg_parser.feature_num = cfg_parser.getint("dataset", "feature_num")
    arg_parser.dataset_name = cfg_parser.get("dataset", "dataset_name")
    arg_parser.dataset_path = os.path.join(
        arg_parser.dataset_base_dir, arg_parser.dataset_name
    )
    arg_parser.eval_every_n_steps = cfg_parser.getint("dataset", "eval_every_n_steps")
    arg_parser.stride = cfg_parser.getint("dataset", "stride", fallback=1)
    arg_parser.handle_missing = cfg_parser.getboolean("dataset", "handle_missing", fallback=True)
    
    # 训练设置
    arg_parser.lr = cfg_parser.getfloat("training", "lr")
    arg_parser.optimizer_type = cfg_parser.get("training", "optimizer_type")
    arg_parser.weight_decay = cfg_parser.getfloat("training", "weight_decay")
    arg_parser.device = cfg_parser.get("training", "device")
    arg_parser.epochs = cfg_parser.getint("training", "epochs")
    arg_parser.early_stop_patience = cfg_parser.getint("training", "early_stop_patience")
    arg_parser.model_saving_strategy = cfg_parser.get("training", "model_saving_strategy")
    arg_parser.max_norm = cfg_parser.getfloat("training", "max_norm")
    
    # 模型设置
    arg_parser.model_name = cfg_parser.get("model", "model_name")
    arg_parser.model_type = cfg_parser.get("model", "model_type")
    
    return arg_parser


def summary_write_into_tb_forecast(summary_writer, info_dict, step, stage):
    """将预测任务的指标写入 TensorBoard"""
    summary_writer.add_scalar(f"total_loss/{stage}", info_dict["total_loss"], step)
    summary_writer.add_scalar(f"forecast_mae/{stage}", info_dict["forecast_mae"], step)
    summary_writer.add_scalar(f"forecast_rmse/{stage}", info_dict["forecast_rmse"], step)
    
    # 如果有重建损失,也记录
    if "reconstruction_loss" in info_dict:
        summary_writer.add_scalar(f"reconstruction_loss/{stage}", info_dict["reconstruction_loss"], step)


def train_forecast(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    summary_writer,
    training_controller,
    logger,
    args
):
    """预测任务的训练循环"""
    logger.info("开始训练预测模型...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = []
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            # 解包数据
            if args.handle_missing:
                history, missing_mask, future = batch_data
                history = history.to(args.device)
                missing_mask = missing_mask.to(args.device)
                future = future.to(args.device)
            else:
                history, future = batch_data
                history = history.to(args.device)
                future = future.to(args.device)
                missing_mask = None
            
            # 前向传播
            optimizer.zero_grad()
            results = model(history, future, missing_mask, stage='train')
            
            # 反向传播
            loss = results['total_loss']
            loss.backward()
            
            # 梯度裁剪
            if args.max_norm != 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
            
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
            # 记录到 TensorBoard
            train_step = training_controller.state_dict["train_step"]
            if train_step % 10 == 0:  # 每10步记录一次
                info_dict = {
                    "total_loss": loss.item(),
                    "forecast_mae": results.get('forecast_loss_mae', torch.tensor(0.0)).item(),
                    "forecast_rmse": torch.sqrt(results.get('forecast_loss_mse', torch.tensor(0.0))).item(),
                }
                if 'reconstruction_loss' in results:
                    info_dict['reconstruction_loss'] = results['reconstruction_loss'].item()
                
                summary_write_into_tb_forecast(summary_writer, info_dict, train_step, "train")
            
            training_controller.state_dict["train_step"] += 1
            
            # 验证
            if train_step % args.eval_every_n_steps == 0:
                state_dict = validate_forecast(
                    model, val_dataloader, summary_writer, training_controller, logger, args
                )
                if state_dict["should_stop"]:
                    logger.info("早停触发,停止训练...")
                    return
                model.train()  # 切回训练模式
        
        # 输出 epoch 统计
        avg_loss = np.mean(epoch_loss)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, 平均损失: {avg_loss:.4f}")
        
        training_controller.state_dict["epoch"] += 1
    
    logger.info("训练完成!")


def validate_forecast(model, val_dataloader, summary_writer, training_controller, logger, args):
    """预测任务的验证"""
    model.eval()
    val_losses = []
    val_maes = []
    val_rmses = []
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            # 解包数据
            if args.handle_missing:
                history, missing_mask, future = batch_data
                history = history.to(args.device)
                missing_mask = missing_mask.to(args.device)
                future = future.to(args.device)
            else:
                history, future = batch_data
                history = history.to(args.device)
                future = future.to(args.device)
                missing_mask = None
            
            # 前向传播
            results = model(history, future, missing_mask, stage='val')
            
            val_losses.append(results['total_loss'].item())
            val_maes.append(results['forecast_loss_mae'].item())
            val_rmses.append(torch.sqrt(results['forecast_loss_mse']).item())
    
    # 计算平均指标
    avg_loss = np.mean(val_losses)
    avg_mae = np.mean(val_maes)
    avg_rmse = np.mean(val_rmses)
    
    logger.info(f"验证 - MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, Loss: {avg_loss:.4f}")
    
    # 记录到 TensorBoard
    info_dict = {
        "total_loss": avg_loss,
        "forecast_mae": avg_mae,
        "forecast_rmse": avg_rmse,
    }
    val_step = training_controller.state_dict["val_step"]
    summary_write_into_tb_forecast(summary_writer, info_dict, val_step, "val")
    
    # 更新训练控制器
    state_dict = training_controller("val", {"imputation_MAE": avg_mae}, logger)
    
    # 保存最佳模型
    if state_dict["save_model"] and args.model_saving_strategy:
        saving_path = os.path.join(
            args.model_saving,
            f"model_trainStep_{state_dict['train_step']}_valStep_{val_step}_MAE_{avg_mae:.4f}.pth"
        )
        save_model(model, optimizer, state_dict, args, saving_path)
        logger.info(f"✅ 模型已保存 -> {saving_path}")
    
    return state_dict


def test_forecast_model(model, test_dataloader, logger, args):
    """测试预测模型"""
    logger.info("开始测试预测模型...")
    model.eval()
    
    test_maes = []
    test_rmses = []
    all_forecasts = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in test_dataloader:
            # 解包数据
            if args.handle_missing:
                history, missing_mask, future = batch_data
                history = history.to(args.device)
                missing_mask = missing_mask.to(args.device)
                future = future.to(args.device)
            else:
                history, future = batch_data
                history = history.to(args.device)
                future = future.to(args.device)
                missing_mask = None
            
            # 预测
            results = model(history, future, missing_mask, stage='test')
            forecast = results['forecast']
            
            # 计算指标
            mae = torch.mean(torch.abs(forecast - future)).item()
            rmse = torch.sqrt(torch.mean((forecast - future) ** 2)).item()
            
            test_maes.append(mae)
            test_rmses.append(rmse)
            
            all_forecasts.append(forecast.cpu().numpy())
            all_targets.append(future.cpu().numpy())
    
    # 汇总结果
    avg_mae = np.mean(test_maes)
    avg_rmse = np.mean(test_rmses)
    
    logger.info("=" * 60)
    logger.info("测试结果:")
    logger.info(f"  MAE:  {avg_mae:.4f}")
    logger.info(f"  RMSE: {avg_rmse:.4f}")
    logger.info("=" * 60)
    
    # 保存结果
    results_path = os.path.join(args.result_saving_path, "forecast_results.npz")
    np.savez(
        results_path,
        forecasts=np.concatenate(all_forecasts, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        mae=avg_mae,
        rmse=avg_rmse
    )
    logger.info(f"✅ 预测结果已保存 -> {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAITS 预测任务训练脚本")
    parser.add_argument("--config_path", type=str, required=True, help="配置文件路径")
    parser.add_argument("--test_mode", action="store_true", help="测试模式")
    parser.add_argument("--param_searching_mode", action="store_true", help="超参搜索模式 (NNI)")
    args = parser.parse_args()
    
    # 检查配置文件
    assert os.path.exists(args.config_path), f'配置文件不存在: "{args.config_path}"'
    
    # 加载配置
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config_path, encoding='utf-8')
    args = read_forecast_arguments(args, cfg)
    
    # 读取模型参数
    args.input_with_mask = cfg.getboolean("model", "input_with_mask")
    args.n_groups = cfg.getint("model", "n_groups")
    args.n_group_inner_layers = cfg.getint("model", "n_group_inner_layers")
    args.param_sharing_strategy = cfg.get("model", "param_sharing_strategy")
    args.d_model = cfg.getint("model", "d_model")
    args.d_inner = cfg.getint("model", "d_inner")
    args.n_head = cfg.getint("model", "n_head")
    args.d_k = cfg.getint("model", "d_k")
    args.d_v = cfg.getint("model", "d_v")
    args.dropout = cfg.getfloat("model", "dropout")
    args.diagonal_attention_mask = cfg.getboolean("model", "diagonal_attention_mask")
    
    # V2 模型的特殊参数
    if args.model_type == "SAITS_Forecaster_V2":
        args.reconstruction_weight = cfg.getfloat("model", "reconstruction_weight", fallback=0.3)
        args.forecast_weight = cfg.getfloat("model", "forecast_weight", fallback=1.0)
    
    # 创建保存目录
    time_now = datetime.now().strftime("%Y-%m-%d_T%H_%M_%S")
    args.model_saving = os.path.join(
        args.result_saving_base_dir, args.model_name, "models", time_now
    )
    args.log_saving = os.path.join(
        args.result_saving_base_dir, args.model_name, "logs", time_now
    )
    os.makedirs(args.model_saving, exist_ok=True)
    os.makedirs(args.log_saving, exist_ok=True)
    
    # 创建日志
    logger = setup_logger(os.path.join(args.log_saving, f"log_{time_now}.log"), "w")
    logger.info(f"配置参数: {args}")
    logger.info(f"配置文件: {args.config_path}")
    logger.info(f"模型名称: {args.model_name}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    forecast_dataloader = ForecastingDataLoader(
        dataset_path=os.path.join(args.dataset_path, "datasets.h5"),
        history_len=args.history_len,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=args.stride,
        handle_missing=args.handle_missing
    )
    
    # 创建模型
    logger.info(f"创建模型: {args.model_type}")
    model_args = {
        "n_groups": args.n_groups,
        "n_group_inner_layers": args.n_group_inner_layers,
        "history_len": args.history_len,
        "forecast_horizon": args.forecast_horizon,
        "d_feature": args.feature_num,
        "d_model": args.d_model,
        "d_inner": args.d_inner,
        "n_head": args.n_head,
        "d_k": args.d_k,
        "d_v": args.d_v,
        "dropout": args.dropout,
        "device": args.device,
        "input_with_mask": args.input_with_mask,
        "diagonal_attention_mask": args.diagonal_attention_mask,
        "param_sharing_strategy": args.param_sharing_strategy,
    }
    
    if args.model_type == "SAITS_Forecaster_V2":
        model_args["reconstruction_weight"] = args.reconstruction_weight
        model_args["forecast_weight"] = args.forecast_weight
    
    model = FORECAST_MODEL_DICT[args.model_type](**model_args)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"可训练参数总数: {total_params:,}")
    
    # 移动到设备
    if "cuda" in args.device and torch.cuda.is_available():
        model = model.to(args.device)
        logger.info(f"✅ 模型已移动到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️ 使用 CPU 训练")
    
    # 测试模式
    if args.test_mode:
        logger.info("=" * 60)
        logger.info("进入测试模式...")
        logger.info("=" * 60)
        
        args.model_path = cfg.get("test", "model_path")
        args.result_saving_path = cfg.get("test", "result_saving_path")
        os.makedirs(args.result_saving_path, exist_ok=True)
        
        # 加载模型
        model = load_model(model, args.model_path, logger)
        
        # 测试
        test_dataloader = forecast_dataloader.get_test_dataloader()
        test_forecast_model(model, test_dataloader, logger, args)
    
    # 训练模式
    else:
        logger.info("=" * 60)
        logger.info("进入训练模式...")
        logger.info("=" * 60)
        
        # 创建优化器
        optimizer = OPTIMIZER[args.optimizer_type](
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        logger.info(f"优化器: {args.optimizer_type}, 学习率: {args.lr}")
        
        # 创建数据加载器
        train_dataloader, val_dataloader = forecast_dataloader.get_train_val_dataloader()
        
        # 创建训练控制器
        training_controller = Controller(args.early_stop_patience)
        
        # 创建 TensorBoard
        tb_writer = SummaryWriter(os.path.join(args.log_saving, f"tensorboard_{time_now}"))
        
        # 开始训练
        train_forecast(
            model,
            optimizer,
            train_dataloader,
            val_dataloader,
            tb_writer,
            training_controller,
            logger,
            args
        )
    
    logger.info("=" * 60)
    logger.info("✅ 所有任务完成!")
    logger.info("=" * 60)
