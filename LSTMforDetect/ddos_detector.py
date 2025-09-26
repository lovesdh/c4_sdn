import pandas as pd
from scapy.all import sniff, rdpcap
from scapy.layers.inet import IP, TCP, UDP
from collections import defaultdict
import numpy as np
import pickle
import os
import time
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 定义类别映射
CLASS_MAP = {'BENIGN': 0, 'DNS': 1, 'LDAP': 2, 'MSSQL': 3, 'NTP': 4,
             'NetBIOS': 5, 'SNMP': 6, 'SSDP': 7, 'Syn': 8, 'TFTP': 9,
             'UDP': 10, 'UDP-lag': 11}
CLASS_NAMES = list(CLASS_MAP.keys())


class Flow:
    """表示网络流的数据结构"""
    __slots__ = ['forward_packets', 'backward_packets', 'packet_count',
                 'proto', 'flow_key', 'start_time', 'end_time']

    def __init__(self):
        self.forward_packets = []  # 上行数据包
        self.backward_packets = []  # 下行数据包
        self.packet_count = 0
        self.proto = 6  # TCP=6,UDP=17
        self.flow_key = None
        self.start_time = float('inf')
        self.end_time = 0

    def add_packet(self, packet, is_upstream: bool):
        """添加数据包到流中"""
        packet_time = packet.time
        if packet_time < self.start_time:
            self.start_time = packet_time
        if packet_time > self.end_time:
            self.end_time = packet_time

        if is_upstream:
            self.forward_packets.append(packet)
        else:
            self.backward_packets.append(packet)
        self.packet_count += 1

    def set_proto(self, flow_key: tuple):
        """设置流协议类型"""
        self.proto = flow_key[4]
        self.flow_key = flow_key

    def _calculate_times(self, packets: list) -> list:
        """计算数据包间到达时间"""
        if len(packets) < 2:
            return []
        return [float(packets[i + 1].time - packets[i].time) for i in range(len(packets) - 1)]

    def _calculate_packet_stats(self, packets: list) -> dict:
        """计算数据包统计特征"""
        if not packets:
            return {
                'total': 0,
                'lengths': [],
                'total_length': 0,
                'max': 0,
                'min': 0,
                'mean': 0,
                'std': 0,
                'header_length': 0
            }

        lengths = [len(pkt) for pkt in packets]
        total_length = sum(lengths)

        # 头部长度计算
        header_lengths = [pkt[IP].ihl * 4 for pkt in packets if IP in pkt]
        header_length = sum(header_lengths) if header_lengths else 0

        return {
            'total': len(packets),
            'lengths': lengths,
            'total_length': total_length,
            'max': max(lengths),
            'min': min(lengths),
            'mean': np.mean(lengths),
            'std': np.std(lengths) if len(lengths) > 1 else 0,
            'header_length': header_length
        }

    def _calculate_tcp_flags(self) -> dict:
        """计算TCP标志统计"""
        flags_count = {
            'fin': 0, 'syn': 0, 'rst': 0, 'psh': 0,
            'ack': 0, 'urg': 0, 'cwe': 0, 'ece': 0
        }

        for pkt in self.forward_packets + self.backward_packets:
            if TCP in pkt:
                flags = pkt[TCP].flags
                if flags & 0x01: flags_count['fin'] += 1
                if flags & 0x02: flags_count['syn'] += 1
                if flags & 0x04: flags_count['rst'] += 1
                if flags & 0x08: flags_count['psh'] += 1
                if flags & 0x10: flags_count['ack'] += 1
                if flags & 0x20: flags_count['urg'] += 1
                if flags & 0x80: flags_count['cwe'] += 1
                if flags & 0x40: flags_count['ece'] += 1

        return flags_count

    def calculate_flow(self) -> Dict[str, Any]:
        """计算流的特征 - 完全匹配训练特征名称"""
        # 1. 基本特征
        flow_duration = self.end_time - self.start_time if self.packet_count > 0 else 0
        inbound = 1 if len(self.forward_packets) < len(self.backward_packets) else 0

        # 2. 全局包统计
        all_packets = self.forward_packets + self.backward_packets
        all_lengths = [len(pkt) for pkt in all_packets]
        if all_lengths:
            min_packet_length = min(all_lengths)
            max_packet_length = max(all_lengths)
            packet_length_mean = np.mean(all_lengths)
            packet_length_std = np.std(all_lengths) if len(all_lengths) > 1 else 0
            average_packet_size = np.mean(all_lengths)
            total_length = sum(all_lengths)
        else:
            min_packet_length = 0
            max_packet_length = 0
            packet_length_mean = 0
            packet_length_std = 0
            average_packet_size = 0
            total_length = 0

        # 3. 流量速率
        if flow_duration > 0:
            flow_packets_per_second = self.packet_count / flow_duration
            flow_bytes_per_second = total_length / flow_duration
        else:
            flow_packets_per_second = 0
            flow_bytes_per_second = 0

        # 4. IAT计算
        forward_iat = self._calculate_times(self.forward_packets)
        backward_iat = self._calculate_times(self.backward_packets)
        all_iat = forward_iat + backward_iat

        # 计算Flow IAT特征
        if all_iat:
            active_mean = np.mean(all_iat)
            active_std = np.std(all_iat)
            active_max = np.max(all_iat)
            active_min = np.min(all_iat)
            flow_iat_mean = np.mean(all_iat)
            flow_iat_std = np.std(all_iat)
            flow_iat_max = np.max(all_iat)
            flow_iat_min = np.min(all_iat)
        else:
            active_mean = 0.0
            active_std = 0.0
            active_max = 0.0
            active_min = 0.0
            flow_iat_mean = 0.0
            flow_iat_std = 0.0
            flow_iat_max = 0.0
            flow_iat_min = 0.0

        # 前向IAT特征
        if forward_iat:
            fwd_iat_total = sum(forward_iat)
            fwd_iat_mean = np.mean(forward_iat)
            fwd_iat_std = np.std(forward_iat)
            fwd_iat_max = np.max(forward_iat)
            fwd_iat_min = np.min(forward_iat)
        else:
            fwd_iat_total = 0.0
            fwd_iat_mean = 0.0
            fwd_iat_std = 0.0
            fwd_iat_max = 0.0
            fwd_iat_min = 0.0

        # 后向IAT特征
        if backward_iat:
            bwd_iat_total = sum(backward_iat)
            bwd_iat_mean = np.mean(backward_iat)
            bwd_iat_std = np.std(backward_iat)
            bwd_iat_max = np.max(backward_iat)
            bwd_iat_min = np.min(backward_iat)
        else:
            bwd_iat_total = 0.0
            bwd_iat_mean = 0.0
            bwd_iat_std = 0.0
            bwd_iat_max = 0.0
            bwd_iat_min = 0.0

        # 5. TCP标志
        tcp_flags = self._calculate_tcp_flags()

        # 6. 前向/后向包统计
        fwd_stats = self._calculate_packet_stats(self.forward_packets)
        bwd_stats = self._calculate_packet_stats(self.backward_packets)

        # 7. 窗口大小
        init_win_fwd = self.forward_packets[0][TCP].window if self.forward_packets and TCP in self.forward_packets[
            0] else 0
        init_win_bwd = self.backward_packets[0][TCP].window if self.backward_packets and TCP in self.backward_packets[
            0] else 0

        # 8. 构建特征字典 - 完全匹配训练时的特征名称
        features = {
            # 基本特征
            ' Protocol': self.proto,
            ' Flow Duration': flow_duration,
            ' Inbound': inbound,

            # 全局包特征
            ' Min Packet Length': min_packet_length,
            ' Packet Length Std': packet_length_std,
            ' Average Packet Size': average_packet_size,
            ' Flow Packets/s': flow_packets_per_second,

            # IAT特征
            ' Flow IAT Max': flow_iat_max,
            'Fwd IAT Total': fwd_iat_total,
            ' Fwd IAT Mean': fwd_iat_mean,
            ' Fwd IAT Std': fwd_iat_std,
            ' Fwd IAT Max': fwd_iat_max,
            ' Fwd IAT Min': fwd_iat_min,
            'Bwd IAT Total': bwd_iat_total,
            ' Bwd IAT Mean': bwd_iat_mean,
            ' Bwd IAT Std': bwd_iat_std,
            ' Bwd IAT Max': bwd_iat_max,
            ' Bwd IAT Min': bwd_iat_min,

            # TCP标志
            ' RST Flag Count': tcp_flags['rst'],
            ' ACK Flag Count': tcp_flags['ack'],
            ' URG Flag Count': tcp_flags['urg'],
            ' CWE Flag Count': tcp_flags['cwe'],
            'Fwd PSH Flags': sum(1 for p in self.forward_packets if TCP in p and p[TCP].flags & 0x08),
            ' Bwd PSH Flags': sum(1 for p in self.backward_packets if TCP in p and p[TCP].flags & 0x20), # Note: This was 0x20, model.py doesn't show this flag calculation explicitly for Bwd PSH. Assuming it's meant to be URG for bwd? Or a typo. Sticking to original for now.

            # 前向流特征
            ' Total Fwd Packets': fwd_stats['total'],
            ' Fwd Packet Length Max': fwd_stats['max'],
            ' Fwd Packet Length Min': fwd_stats['min'],
            ' Fwd Packet Length Mean': fwd_stats['mean'],
            ' Fwd Packet Length Std': fwd_stats['std'],
            ' Fwd Header Length': fwd_stats['header_length'],
            ' Avg Fwd Segment Size': fwd_stats['mean'],
            'Init_Win_bytes_forward': init_win_fwd,
            ' act_data_pkt_fwd': sum(1 for p in self.forward_packets if TCP in p and len(p[TCP].payload) > 0),

            # 后向流特征
            ' Total Backward Packets': bwd_stats['total'],
            'Bwd Packet Length Max': bwd_stats['max'],
            ' Bwd Packet Length Min': bwd_stats['min'],
            ' Bwd Packet Length Mean': bwd_stats['mean'],
            ' Bwd Packet Length Std': bwd_stats['std'],
            ' Bwd Header Length': bwd_stats['header_length'],
            ' Avg Bwd Segment Size': bwd_stats['mean'],
            ' Init_Win_bytes_backward': init_win_bwd,

            # 其他
            'Active Mean': active_mean,
            ' Active Max': active_max,
            ' Active Min': active_min,

            # 元数据
            'Timestamp': time.time(),
            'Flow ID': f"{self.flow_key[0]}_{self.flow_key[1]}_{self.flow_key[2]}_{self.flow_key[3]}_{self.flow_key[4]}"
        }

        return features


def classify_packet(packet, flows: dict, server_ip: str):
    """将数据包分类到对应的流中"""
    if not IP in packet:
        return

    # 确定协议
    if TCP in packet:
        proto = 6
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif UDP in packet:
        proto = 17
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
    else:
        return

    # 创建流键
    flow_key = (packet[IP].src, packet[IP].dst, src_port, dst_port, proto)

    # 确定数据包方向
    is_upstream = packet[IP].dst == server_ip

    # 添加到现有流或创建新流
    if flow_key in flows:
        flows[flow_key].add_packet(packet, is_upstream)
    else:
        new_flow = Flow()
        new_flow.add_packet(packet, is_upstream)
        new_flow.set_proto(flow_key)
        flows[flow_key] = new_flow


def analyze_pcap(pcap_file: str, server_ip: str, max_flows: int = 1000) -> list:
    """分析PCAP文件并提取流特征"""
    flows = {}

    try:
        logger.info(f"开始解析PCAP文件: {pcap_file}")
        packets = rdpcap(pcap_file)
        logger.info(f"成功加载 {len(packets)} 个数据包")

        for i, packet in enumerate(packets):
            classify_packet(packet, flows, server_ip)

            # 定期清理已完成流
            if i % 1000 == 0 and len(flows) > max_flows:
                completed_flows = [k for k, v in flows.items()
                                   if time.time() - v.end_time > 60]
                for k in completed_flows:
                    del flows[k]

    except Exception as e:
        logger.error(f"解析PCAP文件时出错: {str(e)}")
        return []

    # 计算流特征
    flow_features = []
    for flow_key, flow in flows.items():
        if flow.packet_count >= 2:
            try:
                features = flow.calculate_flow()
                features.update({
                    'Source IP': flow_key[0],
                    'Destination IP': flow_key[1],
                    'Source Port': flow_key[2],
                    'Destination Port': flow_key[3]
                })
                flow_features.append(features)
            except Exception as e:
                logger.warning(f"计算流 {flow_key} 特征时出错: {str(e)}")

    logger.info(f"提取了 {len(flow_features)} 个流特征")
    return flow_features


class RNNDetector(torch.nn.Module):
    """支持LSTM/GRU的双向RNN分类器"""

    def __init__(self, input_size=1, hidden_size=128, num_layers=2,
                 num_classes=12, dropout_rate=0.5, bidirectional=True,
                 model_type='gru'):
        super(RNNDetector, self).__init__()
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers # Store num_layers for use in forward pass

        # 选择RNN类型
        if self.model_type == 'lstm':
            self.rnn = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.model_type == 'gru':
            self.rnn = torch.nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}. 选择 'lstm' 或 'gru'")

        # 全连接层 (matching model.py structure)
        rnn_output_size = hidden_size * self.num_directions
        self.batch_norm = torch.nn.BatchNorm1d(rnn_output_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc1 = torch.nn.Linear(rnn_output_size, hidden_size)
        self.relu = torch.nn.GELU() # model.py uses GELU
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        # Note: _init_weights() from model.py is not included here as it's not
        # strictly necessary for loading a state_dict and wasn't in the original ddos_detector.py.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0] # Needed for reshaping hidden state

        # RNN前向传播
        if self.model_type == 'gru':
            # rnn_out: tensor of shape (batch_size, seq_len, num_directions * hidden_size)
            # final_hidden: tensor of shape (num_layers * num_directions, batch_size, hidden_size)
            rnn_out, final_hidden = self.rnn(x)
        elif self.model_type == 'lstm':
            rnn_out, (final_hidden, final_cell_state) = self.rnn(x) # (h_n, c_n)
        else:
             # This case should ideally be caught in __init__
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        # Process final_hidden to get the input for the FC layers (consistent with model.py)
        if self.bidirectional:
            # final_hidden is (num_layers * 2, batch, hidden_size)
            # Reshape to (num_layers, 2, batch, hidden_size) to separate layers and directions
            final_hidden = final_hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            # Get the hidden state from the last layer for forward and backward directions
            final_forward_hidden = final_hidden[-1, 0, :, :]  # Last layer, forward
            final_backward_hidden = final_hidden[-1, 1, :, :] # Last layer, backward
            combined_hidden = torch.cat((final_forward_hidden, final_backward_hidden), dim=1)
        else:
            # final_hidden is (num_layers * 1, batch, hidden_size)
            # Get the hidden state from the last layer (single direction)
            combined_hidden = final_hidden[-1, :, :] # Last layer, single direction

        # Pass through the FC layers as defined in model.py
        out = self.batch_norm(combined_hidden)
        out = self.dropout(out)  # First dropout application
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)  # Second dropout application
        out = self.fc2(out)

        return out


class ModelLoader:
    """模型加载工具类"""

    @staticmethod
    def load_base_model(model_path: str, device: torch.device,
                        model_type: str = 'gru') -> RNNDetector:
        """加载基础RNN模型"""
        # Parameters used here should match those used during training in main.py
        model = RNNDetector(
            input_size=1,        # As per PCA output shape and previous setup
            hidden_size=128,     # Common default, seems consistent
            num_layers=2,        # Common default, seems consistent
            num_classes=len(CLASS_MAP), # Should be 12 based on CLASS_MAP
            dropout_rate=0.5,    # Consistent with main.py training parameters
            bidirectional=True,  # Consistent with main.py training parameters
            model_type=model_type
        )

        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Ensure the state dict keys match the model architecture
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            logger.info(f"成功加载基础模型: {model_path}")
            return model
        except Exception as e:
            logger.error(f"加载基础模型失败: {str(e)}")
            raise

    @staticmethod
    def load_cascade_model(model_path: str, svm_models_dir: str,
                           device: torch.device, confusion_pairs=None) -> 'SVMCascadeModel':
        """加载级联模型"""
        if confusion_pairs is None:
            confusion_pairs = [(10, 11), (5, 7), (2, 8), (9, 10)]

        # 加载基础模型
        base_model = ModelLoader.load_base_model(model_path, device) # model_type can be passed if needed

        # 创建级联模型
        cascade_model = SVMCascadeModel(
            base_model=base_model,
            confusion_pairs=confusion_pairs,
            confidence_threshold=0.95
        )

        # 加载SVM模型
        cascade_model.load_svm_models(svm_models_dir)
        return cascade_model


def preprocess_flow_features(flow_features: list, preprocessor_path: str) -> Tuple[Optional[torch.Tensor], list]:
    """预处理流特征用于模型输入 - 完全匹配训练特征"""
    logger.info("开始预处理流特征...")

    # 创建DataFrame
    df = pd.DataFrame(flow_features)

    # 加载预处理器
    try:
        with open(preprocessor_path, 'rb') as f:
            preprocessors = pickle.load(f)

        # 获取预处理器组件
        encoders = preprocessors.get('encoders', {})
        pca_model = preprocessors.get('pca_model')
        minmax_scaler = preprocessors.get('minmax_scaler')
        numeric_feature_order = preprocessors.get('numeric_feature_order', [])
        base_features = preprocessors.get('base_features', [])
        categorical_features_base = preprocessors.get('categorical_features_base', [])
        label_classes = preprocessors.get('label_classes', CLASS_NAMES)

        # 使用预处理器中的特征顺序
        if not base_features:
            logger.warning("预处理器中未找到特征顺序信息")
            base_features = [ # Fallback, ensure this matches training
                ' Protocol', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
                ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
                'Bwd Packet Length Max', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
                ' Flow Packets/s', ' Flow IAT Max', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
                ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
                ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd Header Length',
                ' Bwd Header Length', ' Min Packet Length', ' Packet Length Std', ' RST Flag Count',
                ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' Average Packet Size',
                ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', 'Init_Win_bytes_forward',
                ' Init_Win_bytes_backward', ' act_data_pkt_fwd', 'Active Mean', ' Active Max',
                ' Active Min', ' Inbound',
            ]


        # 确保所有基础特征存在
        missing_base_features = [f for f in base_features if f not in df.columns]
        if missing_base_features:
            logger.warning(f"填充基础特征中的缺失特征: {missing_base_features}")
            for feature in missing_base_features:
                df[feature] = 0

        # 按基础特征顺序排列
        df = df[base_features]

        # 处理类别特征 - 特别注意' Protocol'特征
        for cat_feature in categorical_features_base:
            if cat_feature in df.columns and cat_feature in encoders:
                encoder = encoders[cat_feature]
                # 进行独热编码
                encoded_data = encoder.transform(df[[cat_feature]])
                encoded_cols = [f"{cat_feature.strip()}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)

                # 删除原始类别特征，并添加编码后的特征
                df = df.drop(cat_feature, axis=1)
                df = pd.concat([df, encoded_df], axis=1)
                logger.info(f"对特征 '{cat_feature}' 进行了独热编码，生成 {len(encoded_cols)} 个新特征")

        # 确保所有数值特征都存在（numeric_feature_order中的特征）
        if not numeric_feature_order:
            logger.warning("预处理器中未找到numeric_feature_order，将使用当前所有数值特征")
            numeric_feature_order = df.select_dtypes(include=['number']).columns.tolist()
        else:
            current_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            # Add missing numeric features expected by scaler/pca and fill with 0
            for feature in numeric_feature_order:
                if feature not in df.columns:
                    logger.warning(f"数值特征 {feature} 缺失，用0填充")
                    df[feature] = 0
            # Ensure order and presence of all columns for scaling
            df = df.reindex(columns=numeric_feature_order, fill_value=0)


    except Exception as e:
        logger.error(f"加载预处理器或特征处理失败: {str(e)}")
        return None, CLASS_NAMES

    # 确保数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 处理缺失值
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    # 归一化
    if minmax_scaler:
        try:
            # Ensure columns match what scaler expects
            # The scaler was likely fit on numeric_feature_order
            df_scaled = pd.DataFrame(minmax_scaler.transform(df[numeric_feature_order]), columns=numeric_feature_order)
            df = df_scaled # df now contains only scaled numeric features in correct order
        except Exception as e:
            logger.error(f"归一化失败: {str(e)}")
            return None, label_classes
    else:
        logger.error("无法应用归一化，归一化器缺失")
        return None, label_classes

    # PCA降维
    if pca_model:
        try:
            # PCA expects data in the same order/shape as during fitting
            pca_result = pca_model.transform(df) # df should now be correctly ordered and scaled
            logger.info(f"PCA降维后特征形状: {pca_result.shape}")
        except Exception as e:
            logger.error(f"PCA降维失败: {str(e)}")
            return None, label_classes
    else:
        logger.error("无法应用PCA降维，PCA模型缺失")
        return None, label_classes

    # 转换为LSTM模型需要的输入格式
    input_tensor = torch.FloatTensor(pca_result).unsqueeze(-1)
    logger.info(f"准备好的模型输入形状: {input_tensor.shape}")

    return input_tensor, label_classes


class SVMCascadeModel:
    """SVM级联分类模型"""

    def __init__(self, base_model: RNNDetector, confusion_pairs: list,
                 confidence_threshold: float = 0.95):
        self.base_model = base_model
        self.confusion_pairs = confusion_pairs
        self.confidence_threshold = confidence_threshold
        self.svm_models = {}

    def load_svm_models(self, model_dir: str):
        """加载SVM二分类器"""
        if not os.path.exists(model_dir):
            logger.warning(f"SVM模型目录不存在: {model_dir}")
            return

        for class1, class2 in self.confusion_pairs:
            model_path = os.path.join(model_dir, f"svm_model_{class1}_{class2}.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        # This assumes SVMModel is defined or pickle can find it
                        self.svm_models[(class1, class2)] = pickle.load(f)
                    logger.info(f"加载SVM分类器: {class1} vs {class2}")
                except Exception as e:
                    logger.error(f"加载SVM模型失败 {model_path}: {str(e)}")
            else:
                logger.warning(f"未找到SVM模型: {model_path}")

    def predict(self, inputs: torch.Tensor, device: torch.device = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用级联模型进行预测"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 基础模型预测
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(inputs.to(device))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            base_preds = outputs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(axis=1)

        # 提取特征 (PCA output before unsqueeze)
        features = inputs.squeeze(-1).cpu().numpy()
        final_preds = base_preds.copy()

        # 应用SVM二次分类
        for i, (pred, conf) in enumerate(zip(base_preds, confidences)):
            if conf < self.confidence_threshold:
                for class1, class2 in self.confusion_pairs:
                    if pred in (class1, class2): # Check if base prediction is one of the pair
                        svm_model_info = self.svm_models.get((class1, class2))
                        if svm_model_info:
                            # Assuming svm_model_info is the actual scikit-learn SVM model
                            # or a wrapper that has a .predict() method
                            # The SVM was trained on PCA features.
                            final_preds[i] = svm_model_info.predict(features[i:i + 1])[0]
                            break # Move to next sample

        return final_preds, base_preds, probs


def predict_flow_type(pcap_file: str, server_ip: str, model_path: str,
                      preprocessor_path: str, svm_models_dir: str = None
                      ) -> List[Dict[str, Any]]:
    """完整的流量分类预测流程"""
    # 1. 提取特征
    logger.info(f"开始从 {pcap_file} 提取流特征...")
    flow_features = analyze_pcap(pcap_file, server_ip)
    if not flow_features:
        logger.error("未能提取任何流特征")
        return []

    # 2. 预处理特征
    logger.info("预处理流特征...")
    result = preprocess_flow_features(flow_features, preprocessor_path)
    # The result unpacking expects Optional[torch.Tensor], list
    if result[0] is None: # Check if tensor is None
        logger.error("特征预处理失败")
        return []

    input_tensor, label_classes = result


    # 3. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        if svm_models_dir and os.path.exists(svm_models_dir): # Check if dir exists
            # Assuming 'gru' is the model_type for the base RNN, adjust if necessary
            model = ModelLoader.load_cascade_model(model_path, svm_models_dir, device, model_type='gru')
            use_cascade = True
        else:
            if svm_models_dir and not os.path.exists(svm_models_dir):
                logger.warning(f"SVM目录 {svm_models_dir} 未找到，仅加载基础模型。")
            # Assuming 'gru' is the model_type, adjust if necessary
            model = ModelLoader.load_base_model(model_path, device, model_type='gru')
            use_cascade = False
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return []

    # 4. 进行预测
    logger.info("进行预测...")
    if input_tensor is None:
        logger.error("输入张量为空，无法进行预测。")
        return[]

    with torch.no_grad():
        inputs = input_tensor.to(device)

        if use_cascade:
            final_preds, base_preds, probs = model.predict(inputs, device)
        else:
            outputs = model(inputs)
            probs_tensor = torch.softmax(outputs, dim=1)
            probs = probs_tensor.cpu().numpy()
            _, predicted_indices = torch.max(probs_tensor, 1)
            final_preds = predicted_indices.cpu().numpy()


    # 5. 整理结果
    results = []
    for i, pred_idx in enumerate(final_preds):
        feature = flow_features[i]
        pred_class = label_classes[pred_idx] if pred_idx < len(label_classes) else f"Unknown-{pred_idx}"

        confidence_value = probs[i, pred_idx]

        results.append({
            'Flow ID': feature.get('Flow ID', f'flow_{i}'),
            'Source IP': feature.get('Source IP', ''),
            'Destination IP': feature.get('Destination IP', ''),
            'Source Port': feature.get('Source Port', ''),
            'Destination Port': feature.get('Destination Port', ''),
            'Protocol': feature.get(' Protocol', ''),
            'Predicted Type': pred_class,
            'Confidence': confidence_value,
            'Duration': feature.get(' Flow Duration', 0)
        })

    # 统计结果
    pred_counts = pd.Series([r['Predicted Type'] for r in results]).value_counts().to_dict()
    logger.info(f"预测类型统计: {pred_counts}")

    return results


# 主函数
def main():
    import argparse

    parser = argparse.ArgumentParser(description='DDoS流量分类预测')
    parser.add_argument('--pcap', default=r"C:\Users\17380\PycharmProjects\LSTMforDetect\UNSW1000.pcap.pcapng",
                        help='PCAP文件路径')
    parser.add_argument('--server_ip', default="149.171.126.3", help='服务器IP地址')
    parser.add_argument('--model', default='./outputs/checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--preprocessor', default='./outputs/preprocessor.pkl', help='预处理器路径')
    parser.add_argument('--svm_dir', default='./outputs/svm_models', help='SVM模型目录（如果使用级联模型）')
    parser.add_argument('--output', default='predictions.csv', help='输出CSV文件路径')

    args = parser.parse_args()

    # 执行预测
    results = predict_flow_type(
        pcap_file=args.pcap,
        server_ip=args.server_ip,
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        svm_models_dir=args.svm_dir
    )

    # 保存结果到CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        logger.info(f"预测结果已保存到 {args.output}")


if __name__ == "__main__":
    main()