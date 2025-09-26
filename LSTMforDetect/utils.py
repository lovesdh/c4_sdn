#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import logging
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import onnx

# 设置matplotlib参数在图表中仍使用英文
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

# 定义类别映射
CLASS_MAP = {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4, 'NetBIOS': 5, 'SNMP': 6, 'SSDP': 7, 'Syn': 8, 'TFTP': 9, 'UDP': 10, 'UDP-lag': 11}
CLASS_NAMES = list(CLASS_MAP.keys())

def evaluate_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """在数据集上评估模型"""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 如果targets是one-hot编码，转换为类别索引
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                target_indices = targets.argmax(dim=1)
            else:
                target_indices = targets.squeeze() if len(targets.shape) > 1 else targets

            outputs = model(inputs)

            loss = criterion(outputs, target_indices)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(target_indices).sum().item()

            y_true.extend(target_indices.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    logger.info(f'评估 | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')

    return avg_loss, accuracy, np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str] = CLASS_NAMES,
                          normalize: bool = True,
                          figsize: Tuple[int, int] = (10, 8),
                          save_path: Optional[str] = None) -> None:
    """
    绘制混淆矩阵

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        normalize: 是否归一化混淆矩阵
        figsize: 图像大小
        save_path: 保存图像的路径，如果为None，则显示图像
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(history: Dict[str, List],
                          figsize: Tuple[int, int] = (12, 5),
                          save_path: Optional[str] = None) -> None:
    """
    绘制训练历史

    参数:
        history: 训练历史字典，包含键:
                'train_loss', 'val_loss', 'train_acc', 'val_acc', 'epochs'
        figsize: 图像大小
        save_path: 保存图像的路径，如果为None，则显示图像
    """
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['epochs'], history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history['epochs'], history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练历史图已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curves(y_true: np.ndarray,
                    y_score: np.ndarray,
                    class_names: List[str] = CLASS_NAMES,
                    figsize: Tuple[int, int] = (12, 10),
                    save_path: Optional[str] = None) -> None:
    """
    绘制多类别分类的ROC曲线

    参数:
        y_true: 真实标签（one-hot编码）
        y_score: 预测分数
        class_names: 类别名称列表
        figsize: 图像大小
        save_path: 保存图像的路径，如果为None，则显示图像
    """
    # 为每个类别计算ROC曲线和ROC面积
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 如果还不是one-hot编码，则进行转换
    if len(y_true.shape) == 1:
        y_true_onehot = np.zeros((y_true.size, n_classes))
        y_true_onehot[np.arange(y_true.size), y_true] = 1
    else:
        y_true_onehot = y_true

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线
    plt.figure(figsize=figsize)

    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC曲线已保存至 {save_path}")
    else:
        plt.show()

    plt.close()


def get_classification_report(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: List[str] = CLASS_NAMES) -> pd.DataFrame:
    """
    获取分类报告作为DataFrame

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表

    返回:
        report_df: 作为DataFrame的分类报告
    """
    # 获取分类报告作为文本
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # 转换为DataFrame
    report_df = pd.DataFrame(report).transpose()

    return report_df


def export_model_onnx(model: torch.nn.Module,
                      sample_input: torch.Tensor,
                      file_path: str) -> None:
    """
    将PyTorch模型导出为ONNX格式

    参数:
        model: PyTorch模型
        sample_input: 具有正确形状的样本输入张量
        file_path: 保存ONNX模型的路径
    """
    model.eval()

    # 导出模型
    torch.onnx.export(
        model,  # 正在运行的模型
        sample_input,  # 模型输入
        file_path,  # 保存模型的位置
        export_params=True,  # 将训练好的参数权重存储在模型文件中
        opset_version=12,  # 导出模型的ONNX版本
        do_constant_folding=True,  # 优化
        input_names=['input'],  # 模型的输入名称
        output_names=['output'],  # 模型的输出名称
        dynamic_axes={
            'input': {0: 'batch_size'},  # 可变长度轴
            'output': {0: 'batch_size'}
        }
    )

    logger.info(f"模型已导出为ONNX格式: {file_path}")

    # 验证模型
    onnx_model = onnx.load(file_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX模型验证通过。")

def softmax(x, axis=None):
    """
    为x中的每组分数计算softmax值。

    参数:
        x: 输入数组
        axis: 计算softmax的轴

    返回:
        softmax_x: x的softmax
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def predict_batch(model: torch.nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  return_probs: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    获取一批数据的预测

    参数:
        model: 训练好的模型
        data_loader: 数据集的DataLoader
        device: 运行预测的设备
        return_probs: 是否返回概率分数

    返回:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率（如果return_probs=True）
    """
    model.eval()

    y_true = []
    y_pred = []
    y_probs = [] if return_probs else None

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if len(targets.shape) > 1 and targets.shape[1] > 1:
                targets = targets.argmax(dim=1)
            else:
                targets = targets.squeeze() if len(targets.shape) > 1 else targets
            outputs = model(inputs)

            _, predicted = outputs.max(1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            if return_probs:
                probs = torch.nn.functional.softmax(outputs, dim=1)
                y_probs.extend(probs.cpu().numpy())

    result = (np.array(y_true), np.array(y_pred))
    if return_probs:
        result += (np.array(y_probs),)

    return result