由于我无法直接生成图片，我将用**PlantUML**代码来表示流程图。您可以将这段代码复制到 [PlantUML在线编辑器](https://www.google.com/search?q=http://www.plantuml.com/plantuml/cref/) 或使用支持PlantUML的工具（如VS Code插件）来生成图形。

------

# DDoS攻击检测系统：项目文档

## 1. 引言

本项目旨在开发一个高效且准确的DDoS（分布式拒绝服务）攻击检测系统。该系统采用**混合级联分类器架构**，结合深度学习（GRU/LSTM）和传统机器学习（XGBoost/SVM）的优势，以应对复杂多变的DDoS攻击。本文档将详细阐述项目的整个Pipeline，包括数据收集、特征选择、数据预处理、模型训练、评估和实时预测等各个关键阶段。

## 2. 项目目标

- 从网络流量数据中准确识别多种类型的DDoS攻击。
- 构建一个两阶段的级联分类器，利用深度学习模型进行初步分类，并使用辅助传统机器学习模型解决易混淆类别的判别问题。
- 提供一套完整的数据处理、模型训练和部署（PCAP预测）的流程。
- 提高DDoS攻击检测的整体准确率和鲁棒性。

## 3. 项目Pipeline总览

根据您的建议，特征选择应在数据预处理（特别是PCA降维）之前进行，以保留原始特征的物理意义并用于相关性分析。整个DDoS攻击检测系统的Pipeline可以分为以下几个主要阶段：

1. **数据收集与初步处理**
2. **特征工程与选择**
3. **数据预处理**
4. 模型训练
   - 主要分类器 (GRU/LSTM) 训练
   - 辅助分类器 (XGBoost/SVM) 训练
5. **模型评估**
6. **实时预测与部署**

下图简要展示了整个流程（PlantUML代码将在末尾提供）：

Code snippet

```
@startuml
skinparam handwritten true
skinparam monochrome true
skinparam shadowing false
skinparam roundcorner 5

rectangle "原始网络流量数据 (PCAP/CSV)" as raw_data

rectangle "1. 数据收集与初步处理" as data_prep {
    component "流量特征提取 (PCAP -> CSV)" as feature_extract
    component "数据加载与合并" as load_merge
    component "数据去重与采样" as dedupe_sample
    raw_data --> feature_extract
    feature_extract --> load_merge
    load_merge --> dedupe_sample
}

rectangle "2. 特征工程与选择" as feature_eng_select {
    component "处理缺失值与无穷值" as handle_na_inf
    component "皮尔逊相关系数分析" as pearson_corr
    component "基于树的特征重要性" as tree_importance
    dedupe_sample --> handle_na_inf
    handle_na_inf --> pearson_corr
    handle_na_inf --> tree_importance
}

rectangle "3. 数据预处理" as data_preproc {
    component "特征筛选" as feature_filter
    component "标签映射与独热编码" as label_encode
    component "特征归一化" as normalize
    component "PCA降维" as pca_dim_reduce
    pearson_corr -down-> feature_filter
    tree_importance -down-> feature_filter
    feature_filter --> label_encode
    label_encode --> normalize
    normalize --> pca_dim_reduce
}

rectangle "4. 模型训练" as model_train {
    component "PyTorch数据集/DataLoader构建" as dataset_loader
    component "主要分类器 (GRU/LSTM) 训练" as primary_train
    component "辅助分类器 (XGBoost/SVM) 训练" as secondary_train
    component "级联模型构建" as cascade_build
    pca_dim_reduce --> dataset_loader
    dataset_loader --> primary_train
    dataset_loader --> secondary_train
    primary_train --> cascade_build
    secondary_train --> cascade_build
}

rectangle "5. 模型评估" as model_eval {
    component "评估指标计算" as metrics_calc
    component "可视化报告" as vis_report
    cascade_build --> metrics_calc
    metrics_calc --> vis_report
}

rectangle "6. 实时预测与部署" as real_time_pred {
    component "实时PCAP解析与特征提取" as real_time_extract
    component "模型加载与预测" as predict_deploy
    component "结果输出" as output_results
    real_time_extract --> predict_deploy
    predict_deploy --> output_results
    cascade_build -down-> predict_deploy : 使用已训练模型
}

@enduml
```

## 4. 详细Pipeline阶段与技术细节

### 4.1. 阶段一：数据收集与初步处理

- **数据来源：** 系统的输入是网络流量数据，通常以PCAP文件或预先提取的CSV文件形式存在。为了训练，我们需要大量的带标签的网络流量数据，其中包含正常流量（BENIGN）和各种DDoS攻击流量（如DNS、LDAP、Syn、UDP等）。

- 流量特征提取（PCAP -> CSV）：

  - 技术细节：

     这一步是将原始的PCAP数据包流转换为具有统计特征的表格数据。

    ```
    ddos_detector.py
    ```

     中的 

    ```
    Flow
    ```

     类和 

    ```
    extract_flow_features
    ```

     函数是这个过程的核心。Scapy库用于解析PCAP文件中的数据包。对于每个TCP/UDP流，会计算一系列统计特征，例如：

    - **时间特征：** `Flow Duration` (流持续时间), `Idle Mean/Max/Min`, `Active Mean/Max/Min` 等。
    - **包数量特征：** `Total Fwd Packets` (前向包总数), `Total Backward Packets` (后向包总数) 等。
    - **字节数特征：** `Total Length of Fwd Packets`, `Total Length of Bwd Packets` 等。
    - **包大小特征：** `Min Packet Length`, `Max Packet Length`, `Average Packet Size`, `Packet Length Std` (包长度标准差) 等。
    - **标志位计数：** `Fwd PSH Flags`, `ACK Flag Count`, `RST Flag Count` 等。
    - **窗口大小：** `Init_Win_bytes_forward`, `Init_Win_bytes_backward`。
    - **流方向：** `Inbound` (入站流量)。

  - **目的：** 将异构的原始网络数据转化为结构化的、可用于机器学习模型的数值特征。

- 数据加载与合并：

  - `CSVSampler` (`sample.py`)：遍历指定文件夹下的所有CSV文件，分块（`chunk_size`）读取数据。这对于处理大型数据集至关重要，可以避免一次性将所有数据加载到内存中。
  - **目的：** 高效地聚合来自不同来源的原始特征数据。

- 数据去重与采样：

  - ```
    CSVSampler
    ```

     (

    ```
    sample.py
    ```

    )：

    - **数据去重：** 通过计算指定列（`target_columns`，例如关键的流标识符和统计特征）的哈希值来识别和去除重复的流量样本。这确保了训练数据的唯一性，避免模型从重复数据中学习到偏差。
    - **下采样/平衡：** 为了解决类别不平衡问题（DDoS数据集中攻击流量通常远少于正常流量），`max_samples_per_class` 参数限制了每个类别的最大样本数。这有助于防止模型过度偏向多数类，提高对少数类攻击的检测能力。

  - **目的：** 确保训练数据的质量、多样性和类别平衡性，提高模型泛化能力。

**涉及文件：**

- `ddos_detector.py`: `Flow` 类和 `extract_flow_features` 函数（提供PCAP特征提取的逻辑）。
- `sample.py`: `CSVSampler` 类，用于大规模CSV文件的采样和去重。

### 4.2. 阶段二：特征工程与选择

此阶段在数据预处理之前进行，旨在识别最具区分度的原始特征，避免引入噪声和冗余，并为后续的归一化和降维提供一个更“干净”的特征集。

- 处理缺失值与无穷值：

  - 在进行特征选择之前，需要对加载的数据进行初步的缺失值（NaN）和无穷大值（`inf`）处理。通常采用填充（如用0、均值或中位数）或删除策略。本项目中通常倾向于用0填充。
  - **目的：** 确保特征的数值有效性，避免后续计算（如相关系数）出错。

- 皮尔逊相关系数分析：

  - 技术细节：

     皮尔逊相关系数（Pearson Correlation Coefficient）衡量两个变量之间的线性相关强度和方向，其值介于-1到1之间。

    - r=fracsum_i=1n(x_i−barx)(y_i−bary)sqrtsum_i=1n(x_i−barx)2sum_i=1n(y_i−bary)2
    - −1lerle1
    - r=1 表示完全正线性相关，r=−1 表示完全负线性相关，r=0 表示无线性相关。

  - 应用：

    - **特征与标签的相关性：** 计算每个特征与目标标签（攻击类型）之间的皮尔逊相关系数。高绝对值的相关系数表明该特征对区分攻击类型有更强的能力。
    - **特征之间的相关性：** 计算特征矩阵内部各特征之间的相关性。如果两个特征之间存在高度相关性（例如，∣r∣0.9），则可能存在冗余。在这种情况下，可以考虑删除其中一个特征，以减少维度并提高模型效率，同时避免共线性问题。

  - **目的：** 识别与目标强相关且彼此之间冗余度低的特征。

- 基于树的特征重要性：

  - 技术细节：

    ```
    NetworkTrafficFeatureSelector
    ```

     (

    ```
    feature_select.py
    ```

    ) 使用基于树的模型，如

    随机森林 (Random Forest)

    、

    Extra Trees

    和

    XGBoost (eXtreme Gradient Boosting)

     来评估特征的重要性。

    - **随机森林/Extra Trees原理：** 这些模型由多棵决策树组成。在构建每棵决策树时，特征在节点分裂（决定最佳分割点）中的使用频率和“纯度增益”（如Gini impurity减少或信息熵减少）可以用来衡量其重要性。一个特征在越多的树中被用于重要的分裂，并且其带来的纯度增益越大，则其重要性越高。

    - XGBoost原理：

       XGBoost也是一个基于树的集成学习算法，它通过迭代地添加弱预测器（决策树）来改进前一个预测器的不足。特征重要性在XGBoost中通常通过三种方式衡量：

      - **`weight` (Frequence):** 特征在所有树中被用作分裂节点的次数。
      - **`gain`:** 特征作为分裂节点带来的平均增益。
      - **`cover`:** 特征作为分裂节点带来的样本覆盖率。

    - **本项目中的应用：** 通常会训练这些模型，然后提取它们的 `feature_importances_` 属性（对于Sklearn模型）或 `get_score(importance_type='gain')` 方法（对于XGBoost）。

  - **目的：** 从模型角度量化每个特征对分类决策的贡献，从而筛选出对模型性能最有益的特征。

**涉及文件：**

- `feature_select.py`: `NetworkTrafficFeatureSelector` 类，用于执行特征选择流程。

### 4.3. 阶段三：数据预处理

在特征选择之后，对筛选出的特征进行进一步的预处理，以适应模型的输入要求并优化训练效果。

- **特征筛选：** 根据皮尔逊相关系数和基于树的特征重要性分析的结果，手动或自动选择一个最优的特征子集。这有助于减少噪声，提高模型效率。

- 标签映射与独热编码：

  - **标签映射：** 将文本标签（如'BENIGN', 'DNS'）映射为数值ID（如0, 1, 2...）。这在 `utils.py` 和 `main.py` 中通过 `CLASS_MAP` 定义。
  - **独热编码 (One-Hot Encoding)：** 对于非数值型或类别型特征（如' Protocol'），将其转换为二进制向量。例如，如果' Protocol'有TCP、UDP、ICMP三个值，独热编码会将其转换为 `[1,0,0]`, `[0,1,0]`, `[0,0,1]`。`DataProcessor` (`data.py`) 使用 `OneHotEncoder` 实现。
  - **目的：** 将非数值型数据转化为数值型，使模型能够处理。

- 特征归一化：

  - 技术细节：

    ```
    DataProcessor
    ```

     (

    ```
    data.py
    ```

    ) 使用 

    ```
    MinMaxScaler
    ```

     对数值特征进行归一化。

    - X_norm=fracX−X_minX_max−X_min
    - 将特征值缩放到0到1之间。这有助于神经网络模型（特别是GRU/LSTM）的稳定训练和收敛，因为不同尺度的特征可能导致梯度不稳定或某些特征对模型更新的影响过大。

  - **目的：** 消除不同特征之间的量纲差异，加速模型收敛，防止少数大数值特征主导模型学习。

- 主成分分析 (PCA) 降维：

  - 技术细节：

    ```
    DataProcessor
    ```

     (

    ```
    data.py
    ```

    ) 使用 

    ```
    PCA
    ```

     (Principal Component Analysis) 对高维特征进行降维。

    - PCA是一种线性降维技术，通过正交变换将原始数据投影到一个新的坐标系中，使得新坐标系中的第一个坐标（主成分）方向上的方差最大，第二个坐标方向上的方差次大，以此类推。
    - 它通过找到数据中方差最大的方向（即主成分）来捕获数据的主要信息。
    - `n_components` 参数指定降维后的维度数量。

  - **目的：** 减少特征维度，降低模型的复杂性，缓解“维度灾难”问题，提高训练速度，并可能降低过拟合风险，同时尽可能保留数据中的重要信息。降维后的特征作为GRU/LSTM模型的输入。

- PyTorch数据集和DataLoader：

  - `DDoSDataset` (`data.py`)：一个自定义的PyTorch `Dataset` 类，用于封装处理后的特征和标签。它负责将Pandas DataFrame或Numpy数组转换为PyTorch张量，并调整维度以适应GRU/LSTM模型的输入要求（`[batch_size, sequence_length, input_size]`）。对于DDoS流量，通常将每个流的特征视为一个时间序列，`sequence_length`可能为1，`input_size`为特征数量。
  - `create_dataloader` (`data.py`)：创建 `DataLoader` 实例，实现数据的批量加载、打乱和并行处理，以提高训练效率。

**涉及文件：**

- `data.py`: `DataProcessor` 类，负责特征工程（标签映射、独热编码、归一化、PCA）和 `DDoSDataset`、`create_dataloader`。
- `utils.py`: 定义了 `CLASS_MAP` 和 `CLASS_NAMES`。

### 4.4. 阶段四：模型训练

此阶段是DDoS检测系统的核心，它构建并训练混合级联分类器。

#### 4.4.1. 主要分类器 (GRU/LSTM) 训练

- 模型架构：

  ```
  RNNDetector
  ```

   (

  ```
  model.py
  ```

  )。尽管命名为

  ```
  RNNDetector
  ```

  ，但它支持选择GRU或LSTM作为核心循环层。

  - **循环神经网络 (RNN) 概述：** RNN是一类专门用于处理序列数据的神经网络。它们通过在网络内部维持一个“隐藏状态”来记忆历史信息，并将其传递给下一个时间步。

  - LSTM (Long Short-Term Memory) 内部结构：

     LSTM是一种特殊的RNN，旨在解决传统RNN的梯度消失/爆炸问题，能够学习长期依赖关系。它通过引入“门”结构（输入门、遗忘门、输出门）来控制信息在细胞状态（cell state）中的流动，从而选择性地记住或遗忘信息。

    - **遗忘门 (f_t):** 决定从上一个隐藏状态 h_t−1 和当前输入 x_t 中，应该“遗忘”哪些信息。
    - **输入门 (i_t):** 决定哪些新信息会存储到细胞状态中。
    - **候选细胞状态 (tildeC_t):** 根据当前输入和上一隐藏状态计算的新的候选信息。
    - **细胞状态更新 (C_t):** 结合遗忘门和输入门来更新细胞状态。
    - **输出门 (o_t):** 决定当前细胞状态中哪些信息将输出到隐藏状态 h_t。

  - GRU (Gated Recurrent Unit) 内部结构：

     GRU是LSTM的一个简化版本，拥有更少的门（更新门和重置门），但通常也能达到相似的性能，且计算效率更高。

    - **更新门 (z_t):** 决定有多少过去的信息需要继续传递到未来，以及有多少新的信息需要被采纳。
    - **重置门 (r_t):** 决定如何将过去的信息与新的输入结合。
    - **候选隐藏状态 (tildeh_t):** 类似于LSTM的候选细胞状态，用于存储当前时间步的潜在信息。
    - **隐藏状态更新 (h_t):** 根据更新门和重置门来更新隐藏状态。

  - **双向 (Bidirectional) RNN：** 模型支持双向结构，这意味着它同时处理输入序列的正向和反向信息。这使得模型能够同时捕获序列的过去和未来的上下文信息，对于流量数据这种具有时序依赖性但不严格方向性的数据非常有用。

  - **层数与Dropout：** 通常包含多层GRU/LSTM（`num_layers`），并在层之间或输出层之前加入Dropout层（`dropout_rate`）以防止过拟合。最终的全连接层用于将循环层的输出映射到多类别分类的输出。

- 训练器：

  ```
  LSTMTrainer
  ```

   (

  ```
  trainer.py
  ```

  )。

  - **损失函数：** 使用交叉熵损失 (`CrossEntropyLoss`)，适用于多类别分类任务。
  - **优化器：** 通常使用Adam或AdamW等自适应学习率优化器，能够高效地更新模型权重。
  - **学习率调度：** 可能包含学习率衰减策略（例如，余弦退火或ReduceLROnPlateau），以在训练过程中动态调整学习率，帮助模型更好地收敛。
  - **梯度裁剪：** 对于RNN模型，为了防止训练过程中出现梯度爆炸（梯度值变得非常大，导致模型权重更新不稳定），通常会使用梯度裁剪 (`gradient_clip_val`) 来限制梯度的最大范数。
  - **模型保存：** 在验证集上表现最佳的模型将被保存，以供后续使用。

- **训练流程：** `train_lstm_model` (`main.py`) 负责协调这个过程，包括数据加载、模型初始化、训练循环和模型保存。

#### 4.4.2. 辅助分类器 (XGBoost/SVM) 训练

主要分类器在某些类别之间可能存在混淆。辅助分类器专门用于解决这些“模糊”情况。

- **识别混淆类别对：** 在 `main.py` 中定义 `confusion_pairs`，这些对是根据经验、领域知识或初步模型评估（如混淆矩阵分析）结果确定的容易混淆的类别。例如，某些DDoS攻击类型可能在流量特征上非常相似。
- 模型架构：
  - XGBoost (eXtreme Gradient Boosting)：
    - **技术细节：** XGBoost是一个高度优化且可扩展的梯度提升决策树系统。它通过迭代地训练一系列弱学习器（通常是决策树），并将它们的预测结果累加起来形成最终的强学习器。每次迭代都会尝试纠正前一棵树的错误（通过最小化损失函数的梯度）。
    - **优势：** 具有出色的性能、速度和处理大规模数据的能力，并且能自动处理缺失值。
  - SVM (Support Vector Machine)：
    - **技术细节：** SVM是一种二分类模型，其基本思想是找到一个超平面，能够最大化地将不同类别的样本分开，同时使支持向量（离超平面最近的样本）到超平面的距离最大化（间隔最大化）。对于非线性可分数据，SVM通过核函数（如径向基函数RBF）将数据映射到高维空间，使其在高维空间中变得线性可分。
    - **优势：** 在处理中小规模数据集时表现良好，对高维数据具有鲁棒性，且泛化能力强。
- 训练器：
  - `XGBoostTrainer` (`trainer.py`)：负责训练XGBoost模型。它会针对每个混淆类别对，从原始数据中提取只包含这两个类别的子集，并训练一个二分类器。
  - `SVMTrainer` (`trainer.py`)：类似地，训练SVM模型。
  - **超参数调优 (GridSearchCV)：** `XGBoostTrainer` 和 `SVMTrainer` 都可能使用 `GridSearchCV` 进行超参数优化。这是通过在预定义的超参数网格中遍历所有可能的组合，并使用交叉验证来评估每个组合的性能，最终选择性能最佳的超参数集。这确保了每个辅助分类器都能达到最佳性能。

**涉及文件：**

- `model.py`: 定义 `RNNDetector` (GRU/LSTM), `XGBoostModel`, `XGBoostCascadeModel`, `SVMModel`, `SVMCascadeModel`。
- `trainer.py`: 定义 `LSTMTrainer`, `XGBoostTrainer`, `SVMTrainer`。
- `main.py`: 协调整个训练过程，包括选择级联模型类型 (`CASCADE_MODEL_TYPE`) 和定义 `confusion_pairs`。

### 4.5. 阶段五：模型评估

在模型训练完成后，需要对模型性能进行全面评估，以了解其在未见过数据上的表现。

- 评估指标：
  - **准确率 (Accuracy):** 整体分类正确的样本比例。
  - **精确率 (Precision):** 预测为正类中，真正为正类的比例。
  - **召回率 (Recall):** 真实为正类中，被预测为正类的比例。
  - **F1-Score:** 精确率和召回率的调和平均值。
  - **混淆矩阵 (Confusion Matrix):** 直观展示各类别之间的分类正确性和错误情况，特别是识别哪些类别容易被混淆。
  - **分类报告 (Classification Report):** 包含每个类别的精确率、召回率、F1-Score和支持数，提供详细的类别级别性能分析。
  - **ROC曲线和AUC (Receiver Operating Characteristic & Area Under the Curve):** 衡量二分类模型在不同阈值下的表现。对于多分类问题，通常进行One-vs-Rest（OVR）或One-vs-One（OVO）的ROC分析。AUC值越大，模型性能越好。
- 评估工具：
  - `evaluate_model` (`utils.py`)：用于在给定数据集上获取模型的预测结果。
  - `plot_confusion_matrix`, `plot_roc_curve`, `plot_precision_recall_curve` (`utils.py`)：用于生成各种评估指标的可视化图表。
  - `print_classification_report` (`utils.py`)：打印详细的分类性能报告。

**涉及文件：**

- `utils.py`: 包含模型评估和可视化相关的所有函数。

### 4.6. 阶段六：实时预测与部署

这是DDoS检测系统投入实际使用的阶段，它将训练好的模型应用于实时的网络流量。

- **PCAP文件处理：** `ddos_detector.py` 能够读取PCAP文件，从中解析网络数据包。

- 流量重组与特征提取：

  - `Flow` 类 (`ddos_detector.py`)：用于存储属于同一网络流的数据包（根据源/目的IP、端口和协议进行定义）。这类似于训练数据预处理时的特征提取过程。
  - `process_packet` (`ddos_detector.py`)：实时处理单个数据包，将其分配到相应的流中，并更新流的统计信息。
  - `extract_flow_features` (`ddos_detector.py`)：从完整或部分构建的流中计算出与训练数据相同的特征。

- **实时预处理：** 在PCAP中提取的流量特征需要经过与训练数据相同的预处理步骤（归一化、PCA）。这通过加载在训练阶段保存的 `DataProcessor` 对象（其中包含了训练时学习到的`MinMaxScaler`和`PCA`模型）来实现，确保数据转换的一致性。

- 级联模型最终判断（投票融合方式 - 设想）：

  - **基础预测：** 首先，原始流量数据（经过预处理）输入到主GRU/LSTM模型中，得到初步的分类结果和每个类别的置信度分数（概率分布）。

  - **置信度判断与辅助触发：** 对于每个样本，如果主模型的预测置信度（最高概率）低于某个预设的阈值（例如0.8），并且其预测类别属于预定义的 `confusion_pairs` 之一，则触发辅助分类器。

  - **辅助分类器投票：** 当一个样本被传递给辅助分类器时，它将被送到所有相关的辅助分类器（例如，如果主模型预测为类别A，而A与B、C构成混淆对，则样本会被送入A-B和A-C的二分类器）。每个辅助分类器会给出一个二分类结果（例如，是A还是B？是A还是C？）。

  - 投票融合：

     级联模型收集所有被触发的辅助分类器的预测结果，以及主模型的初步预测。

    - **策略一（权重投票）：** 可以给主模型较高的权重，给辅助模型较低但有影响力的权重。例如，如果主模型倾向于A，辅助A-B模型倾向于B，则可以根据权重加权投票。
    - **策略二（多数投票）：** 简单地选择投票最多的类别作为最终预测。
    - **策略三（优先级回退）：** 如果辅助模型之间达成一致，则采纳辅助模型的预测；如果辅助模型之间意见不一，或者主模型置信度很高，则可能回退到主模型的预测。

  - **本项目代码中的实现：** **当前代码中，级联判断逻辑是如果基础模型预测的置信度低于阈值，并且预测类别属于某个混淆对，则直接将样本的特征交给该混淆对对应的辅助分类器（XGBoost/SVM）进行重新预测，并将其预测结果作为最终结果。它不是一个多模型投票融合，而是基于置信度的二阶段决策。** 我在此文档中按您的要求阐述了“投票融合”的设想，这可以作为未来的优化方向。

- **结果输出：** 最终的预测结果（包括原始流信息、预测标签、置信度等）可以输出到CSV文件或其他格式，便于分析和后续处理。

**涉及文件：**

- `ddos_detector.py`: 负责PCAP文件解析、流量重组、特征提取、实时预处理和模型预测。
- `data.py`: `DataProcessor` 类，其实例在预测阶段被加载用于数据预处理。
- `model.py`: `BiLSTMDetector` 和 `XGBoostCascadeModel` (或 `SVMCascadeModel`) 在预测阶段被加载和使用。

## 5. 日志与输出

- **日志：** 项目通过 `logging` 模块记录关键操作和信息，日志文件 `ddos_detection.log` 会记录训练进度、评估结果和预测信息。这对于调试和追踪系统行为至关重要。
- 输出目录 (`./outputs`):
  - `checkpoints/`: 保存主要GRU/LSTM模型的最佳权重，以`.pth`文件形式存在。
  - `preprocessor.pkl`: 保存 `DataProcessor` 实例，其中包含了训练时学习到的 `MinMaxScaler`、`OneHotEncoder` 和 `PCA` 模型，用于保证未来预测时数据预处理的一致性。
  - `xgb_models/` 或 `svm_models/`: 保存训练好的辅助分类器（XGBoost或SVM）模型，每个混淆对对应一个模型文件。
  - 其他可能的输出包括评估图表（混淆矩阵、ROC曲线等）和详细的分类报告。

## 6. 使用说明

1. 环境配置：

    确保您的Python环境已安装所有必要的库。建议使用 

   ```
   pip
   ```

    进行安装：

   Bash

   ```
   pip install torch scikit-learn pandas numpy matplotlib seaborn xgboost scapy tqdm
   ```

2. 数据准备：

   - 准备您的原始网络流量CSV文件。这些文件应包含网络流量特征和名为 `' Label'` 的标签列。

   - 修改 `sample.py` 中的 `folders` 和 `output_folder` 变量，使其指向您的原始数据路径和采样数据保存路径。

   - 运行 

     ```
     sample.py
     ```

      对数据进行初步处理、去重和采样：

     Bash

     ```
     python sample.py
     ```

   - （重要）

      运行 

     ```
     feature_select.py
     ```

      进行特征选择。这将帮助您识别最重要的特征，并建议您用于后续模型的特征子集。

     Bash

     ```
     python feature_select.py --csv_path /path/to/your/sampled_data.csv
     ```

     根据

     ```
     feature_select.py
     ```

     的输出，您可能需要更新

     ```
     data.py
     ```

     中

     ```
     DataProcessor
     ```

     的

     ```
     base_features
     ```

     列表，以包含经过筛选的特征。

3. 训练模型：

   - 修改 `main.py` 中的 `train_data_path`, `val_data_path` (指向经过采样和特征选择后的CSV文件), `output_dir` (保存模型和预处理器的目录) 和 `CASCADE_MODEL_TYPE` (选择 "xgboost" 或 "svm")。

   - 调整 `main.py` 中的 `confusion_pairs` 列表，以反映您数据集中最容易混淆的类别。

   - 运行训练脚本：

     Bash

     ```
     python main.py
     ```

4. 实时检测：

   - 确保您的PCAP文件路径、训练好的GRU/LSTM模型路径、数据预处理器路径以及辅助分类器模型目录在 `ddos_detector.py` 的参数中设置正确。

   - 运行实时预测脚本：

     Bash

     ```
     python ddos_detector.py --pcap /path/to/your/traffic.pcap --server_ip YOUR_SERVER_IP --model ./outputs/checkpoints/best_model.pth --preprocessor ./outputs/preprocessor.pkl --svm_dir ./outputs/svm_models --output predictions.csv
     # 或如果使用XGBoost级联模型
     # python ddos_detector.py --pcap /path/to/your/traffic.pcap --server_ip YOUR_SERVER_IP --model ./outputs/checkpoints/best_model.pth --preprocessor ./outputs/preprocessor.pkl --xgb_dir ./outputs/xgb_models --output predictions.csv
     ```

     ```
     YOUR_SERVER_IP
     ```

      替换为您实际的服务器IP地址，以便正确识别流量方向。

## 8. 结论

这个DDoS攻击检测系统提供了一个全面且模块化的解决方案，从原始网络流量数据到实时攻击检测。其混合级联模型的设计旨在平衡深度学习的强大特征学习能力和传统机器学习在特定问题上的精准判别力，从而实现高准确率的DDoS攻击检测。通过详细的日志记录和灵活的配置选项，本系统易于理解、扩展和部署。

------

**PlantUML 流程图代码：**

Code snippet

```
@startuml
skinparam handwritten true
skinparam monochrome true
skinparam shadowing false
skinparam roundcorner 5

rectangle "原始网络流量数据 (PCAP/CSV)" as raw_data

rectangle "1. 数据收集与初步处理" as data_prep {
    component "流量特征提取 (PCAP -> CSV)" as feature_extract
    component "数据加载与合并" as load_merge
    component "数据去重与采样" as dedupe_sample
    raw_data --> feature_extract
    feature_extract --> load_merge
    load_merge --> dedupe_sample
}

rectangle "2. 特征工程与选择" as feature_eng_select {
    component "处理缺失值与无穷值" as handle_na_inf
    component "皮尔逊相关系数分析" as pearson_corr
    component "基于树的特征重要性" as tree_importance
    dedupe_sample --> handle_na_inf
    handle_na_inf --> pearson_corr
    handle_na_inf --> tree_importance
}

rectangle "3. 数据预处理" as data_preproc {
    component "特征筛选" as feature_filter
    component "标签映射与独热编码" as label_encode
    component "特征归一化" as normalize
    component "PCA降维" as pca_dim_reduce
    pearson_corr -down-> feature_filter
    tree_importance -down-> feature_filter
    feature_filter --> label_encode
    label_encode --> normalize
    normalize --> pca_dim_reduce
}

rectangle "4. 模型训练" as model_train {
    component "PyTorch数据集/DataLoader构建" as dataset_loader
    component "主要分类器 (GRU/LSTM) 训练" as primary_train
    component "辅助分类器 (XGBoost/SVM) 训练" as secondary_train
    component "级联模型构建" as cascade_build
    pca_dim_reduce --> dataset_loader
    dataset_loader --> primary_train
    dataset_loader --> secondary_train
    primary_train --> cascade_build
    secondary_train --> cascade_build
}

rectangle "5. 模型评估" as model_eval {
    component "评估指标计算" as metrics_calc
    component "可视化报告" as vis_report
    cascade_build --> metrics_calc
    metrics_calc --> vis_report
}

rectangle "6. 实时预测与部署" as real_time_pred {
    component "实时PCAP解析与特征提取" as real_time_extract
    component "模型加载与预测" as predict_deploy
    component "结果输出" as output_results
    real_time_extract --> predict_deploy
    predict_deploy --> output_results
    cascade_build -down-> predict_deploy : 使用已训练模型
}

@enduml
```