#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import logging
import sys
from collections import Counter

sys.path.append(".")  # 确保可以导入当前目录下的模块
from model import RNNDetector, SVMCascadeModel, XGBoostCascadeModel  # 添加XGBoostCascadeModel
from data import DDoSDataset  # 只导入DDoSDataset
from torch.utils.data import DataLoader  # 直接导入PyTorch的DataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 定义类别映射
CLASS_MAP = {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4, 'NetBIOS': 5,
             'SNMP': 6, 'SSDP': 7, 'Syn': 8, 'TFTP': 9, 'UDP': 10, 'UDP-lag': 11}
REVERSE_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """创建数据加载器 - 本地实现"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def predict_with_cascade_model(csv_path, output_path=None,
                               preprocessor_path="./outputs/preprocessor.pkl",
                               lstm_model_path="./outputs/checkpoints/best_model.pth",
                               svm_models_dir="./outputs/svm_models",
                               xgb_models_dir="./outputs/xgb_models",  # 新增参数
                               model_type="svm",  # 新增参数，选择使用SVM还是XGBoost
                               confusion_pairs=None):
    """
    使用级联模型进行预测

    参数:
        csv_path: 要预测的CSV文件路径
        output_path: 输出结果的CSV文件路径
        preprocessor_path: 预处理器路径
        lstm_model_path: LSTM模型路径
        svm_models_dir: SVM模型目录
        xgb_models_dir: XGBoost模型目录
        model_type: 选择使用的级联模型类型 ("svm" 或 "xgboost")
        confusion_pairs: 混淆类别对列表

    返回:
        results: 预测结果
    """
    if confusion_pairs is None:
        confusion_pairs = [(10, 11), (5, 7), (2, 8), (9, 10), (0, 4), (0, 9)]  # 更新混淆对

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    logger.info(f"开始处理CSV文件: {csv_path}")
    try:
        test_dataset = DDoSDataset(
            data_path=csv_path,
            preprocessor_path=preprocessor_path,
            train=False
        )
        logger.info(f"数据集大小: {len(test_dataset)}")

        test_loader = create_dataloader(
            dataset=test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )

        try:
            flow_ids = test_dataset.processor.load_data().get('Flow ID', None)
            if flow_ids is None:
                logger.warning("无法获取Flow ID")
                flow_ids = [f"flow_{i}" for i in range(len(test_dataset))]
        except Exception as e:
            logger.warning(f"获取Flow ID时出错: {e}")
            flow_ids = [f"flow_{i}" for i in range(len(test_dataset))]

    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        return None

    logger.info(f"加载基础模型: {lstm_model_path}")
    try:
        checkpoint = torch.load(lstm_model_path, map_location=device)

        base_model = RNNDetector(
            input_size=1,
            hidden_size=128,  # 更新
            num_layers=2,
            num_classes=12,   # 更新：从13改为12
            dropout_rate=0.5,
            bidirectional=True,
            model_type='gru'  # 新增：使用GRU
        )
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_model.to(device)
        base_model.eval()
    except Exception as e:
        logger.error(f"加载基础模型时出错: {e}")
        return None

    # 创建级联模型 - 支持选择不同类型
    logger.info(f"创建{model_type.upper()}级联模型...")
    if model_type.lower() == "xgboost":
        cascade_model = XGBoostCascadeModel(
            base_model=base_model,
            confusion_pairs=confusion_pairs,
            confidence_threshold=0.95
        )
        # 加载XGBoost模型
        cascade_model.load_xgb_models(xgb_models_dir)
    else:  # 默认使用SVM
        cascade_model = SVMCascadeModel(
            base_model=base_model,
            confusion_pairs=confusion_pairs,
            confidence_threshold=0.95
        )
        # 加载SVM模型
        cascade_model.load_svm_models(svm_models_dir)

    logger.info("开始预测...")
    predictions = {
        'flow_id': flow_ids,
        'base_prediction': [],  # 改名以更通用
        'base_confidence': [],
        'final_prediction': [],
        'secondary_used': [],   # 改名以更通用
        'prediction_label': []
    }

    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            final_pred, base_pred, probs = cascade_model.predict(inputs, device)
            if isinstance(probs, np.ndarray):
                confidence = probs.max(axis=1)
            else:
                confidence = probs.max(1)[0].cpu().numpy()

            batch_size = inputs.size(0) if hasattr(inputs, 'size') else len(inputs)
            secondary_used = np.zeros(batch_size, dtype=bool)

            for j, (pred, conf) in enumerate(zip(base_pred, confidence)):
                for class1, class2 in confusion_pairs:
                    if pred in [class1, class2] and conf < 0.95:
                        if j < len(final_pred) and final_pred[j] != base_pred[j]:
                            secondary_used[j] = True
                        break

            predictions['base_prediction'].extend(base_pred)
            predictions['base_confidence'].extend(confidence)
            predictions['final_prediction'].extend(final_pred)
            predictions['secondary_used'].extend(secondary_used)
            predictions['prediction_label'].extend([REVERSE_CLASS_MAP.get(p, f"Unknown-{p}") for p in final_pred])

            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                logger.info(f"批次: {i + 1}/{len(test_loader)}")

    min_len = min(len(arr) for arr in predictions.values())
    for key in predictions:
        predictions[key] = predictions[key][:min_len]

    base_counter = Counter(predictions['base_prediction'])
    final_counter = Counter(predictions['final_prediction'])
    secondary_used_count = sum(predictions['secondary_used'])

    logger.info(f"基础模型预测分布: {dict(base_counter)}")
    logger.info(f"最终预测分布: {dict(final_counter)}")
    logger.info(f"使用{model_type.upper()}次数: {secondary_used_count}/{len(predictions['base_prediction'])} ({secondary_used_count / len(predictions['base_prediction']) * 100:.2f}%)")

    for class1, class2 in confusion_pairs:
        base_class1 = sum(1 for p in predictions['base_prediction'] if p == class1)
        base_class2 = sum(1 for p in predictions['base_prediction'] if p == class2)

        final_class1 = sum(1 for p in predictions['final_prediction'] if p == class1)
        final_class2 = sum(1 for p in predictions['final_prediction'] if p == class2)

        change_class1 = final_class1 - base_class1
        change_class2 = final_class2 - base_class2

        if base_class1 > 0 or base_class2 > 0:
            logger.info(f"类别 {REVERSE_CLASS_MAP.get(class1, class1)} vs {REVERSE_CLASS_MAP.get(class2, class2)}:")
            logger.info(f"  基础模型: {base_class1}/{base_class2}")
            logger.info(f"  最终结果: {final_class1}/{final_class2}")
            logger.info(f"  变化: {change_class1}/{change_class2}")

    if output_path:
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        logger.info(f"预测结果已保存到: {output_path}")

    return predictions


def main():
    import argparse

    parser = argparse.ArgumentParser(description='DDoS流量级联预测')
    parser.add_argument('--csv', default="C:\\Users\\17380\\test_dataset.csv", help='需要预测的CSV文件路径')
    parser.add_argument('--output', default='prediction_results.csv', help='输出结果CSV文件路径')
    parser.add_argument('--preprocessor', default='./outputs/preprocessor.pkl', help='预处理器路径')
    parser.add_argument('--model', default='./outputs/checkpoints/best_model.pth', help='基础模型路径')  # 改名
    parser.add_argument('--svm_dir', default='./outputs/svm_models', help='SVM模型目录')
    parser.add_argument('--xgb_dir', default='./outputs/xgb_models', help='XGBoost模型目录')  # 新增
    parser.add_argument('--model_type', default='xgboost', choices=['svm', 'xgboost'], help='级联模型类型')  # 新增

    args = parser.parse_args()

    predict_with_cascade_model(
        csv_path=args.csv,
        output_path=args.output,
        preprocessor_path=args.preprocessor,
        lstm_model_path=args.model,  # 改名
        svm_models_dir=args.svm_dir,
        xgb_models_dir=args.xgb_dir,  # 新增
        model_type=args.model_type   # 新增
    )


if __name__ == "__main__":
    main()