# 分步实施提示词手册 (Step-by-Step Implementation Prompts Playbook)

> **文档定位**：本手册为《梅尔谱 CNN 语音任务中的注意力机制探索》项目提供标准化的分步开发与执行提示词。每个 Step 独立自洽，包含明确的任务目标、代码实现规范、输入输出格式与验收单测。

---

## 阶段一：纯净数据流与时频前端构建 (Phase 1)

### 提示词 1.1：纯 Torch 梅尔谱时频提取模块
```markdown
【任务名称】构建纯 PyTorch 零外部编译依赖的 Log-Mel 谱图前端模块
【目标文件】`src/data/audio_frontend.py` 与 `tests/test_frontend.py`

【技术规范与数学要求】
1. 严禁直接依赖 torchaudio（避免 Windows C++ 底层动态库兼容问题），基于 `torch.stft` 与手写 Slaney 归一化三角滤波器组实现。
2. 音频预处理：
   - 统一采样率：16000 Hz（如遇非 16k 音频，使用 scipy.signal.resample_poly 进行抗混叠多相滤波）。
   - STFT 参数：n_fft=1024, win_length=1024, hop_length=256, window=torch.hann_window。
   - 梅尔频带：n_mels=128, f_min=0.0, f_max=8000.0。
   - 对数压缩：Log-Mel = torch.log(mel_power + 1e-6)。
3. 时域长度归一化：
   - 目标帧数固定为 fixed_t = 300 帧（约 4.8 秒）。
   - 若实际帧数 < 300，采用模式反射填充（Reflect Pad）或零填充；若 > 300，采用中心对称截断（Center Crop）。
4. 差分扩展（3 通道）：
   - 通道 0：静态 Log-Mel 能量；
   - 通道 1：一阶时域差分 Δ (Delta, 速度特征)；
   - 通道 2：二阶时域差分 ΔΔ (Delta-Delta, 加速度特征)；
   - 最终输出张量形状严格为 `(B, 3, 128, 300)`。

【验收单元测试要求】
- 编写 `tests/test_frontend.py`：
  - 测试随机波形张量 `(4, 48000)` 前向通过后输出张量形状为 `(4, 3, 128, 300)`；
  - 测试极短音频（如 0.5 秒）和极长音频（如 8 秒）均能稳定输出 `(1, 3, 128, 300)` 且无 NaN / Inf。
```

---

### 提示词 1.2：说话人无关 5 折无泄露划分与折内标准化
```markdown
【任务名称】构建严密无泄露的说话人分组 5 折交叉验证划分器与 Dataset
【目标文件】`src/data/split.py`, `src/data/dataset.py` 与 `tests/test_leakage.py`

【技术规范与学术防泄露红线】
1. 数据集：RAVDESS（24 名演员，每人 60 条，共 1440 条 wav，8 类情绪：0:neutral, 1:calm, 2:happy, 3:sad, 4:angry, 5:fearful, 6:disgust, 7:surprised）。
2. 5 折划分算法：
   - 使用 `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`。
   - 分组 Group 键：演员编号 Actor ID（1 ~ 24）。
   - 分层 Stratify 键：情绪标签（0 ~ 7）。
   - 将划分结果持久化保存为 `configs/folds.json`（记录每折的 train_actors, val_actors, train_files, val_files）。
3. 严格防泄露断言：
   - 任意两折的验证集演员交集必须为空：`set(val_actors_i) ∩ set(val_actors_j) == ∅`。
   - 当前折训练集演员与验证集演员交集必须为空：`set(train_actors_i) ∩ set(val_actors_i) == ∅`。
4. 折内标准化（In-Fold Normalization）：
   - 计算均值与标准差时，仅统计当前折训练集样本的梅尔谱：$\mu_i, \sigma_i$。
   - 验证集样本必须严格使用当前折训练集的 $\mu_i, \sigma_i$ 进行归一化，严禁全量全局标准化。

【验收单元测试要求】
- 编写 `tests/test_leakage.py`，执行硬断言检查 `configs/folds.json` 的 5 折演员集合无任何交集。
```

---

## 阶段二：注意力算子库与统一骨干网络 (Phase 2)

### 提示词 2.1：模块化注意力算子实现
```markdown
【任务名称】实现解耦的通道注意力 (SE)、时域多头注意力池化 (MHAP) 与统计池化 (ASP)
【目标文件】`src/models/blocks.py` 与 `tests/test_attention_blocks.py`

【技术规范与算子细节】
1. `SELayer(channel, reduction=16)`：
   - 机制：GAP(X) -> Linear(C, C//16) -> ReLU -> Linear(C//16, C) -> Sigmoid -> 逐通道乘回。
   - 输入张量：`(B, C, F, T)`，输出张量：`(B, C, F, T)`。
2. `MultiHeadAttentivePooling(in_dim=256, n_heads=4, hidden_dim=128)`：
   - 机制：针对时间轴做多头加权汇聚。
   - 对每个 Head：$u_t = \tanh(W x_t + b)$，$\alpha_t = \text{softmax}(v^\top u_t)$，$c = \sum \alpha_t u_t$。
   - 多头拼接后经线性投影 $W_{\text{out}}$ 输出回 `in_dim` 维度。
   - 输入张量：`(B, T, in_dim)`，输出：`pooled: (B, in_dim)`，可选返回 `attn_maps: (B, n_heads, T)`。
3. `AttentiveStatisticsPooling(in_dim=256, hidden_dim=128)`：
   - 机制：计算加权均值 $\mu = \sum \alpha_t x_t$ 与加权标准差 $\sigma = \sqrt{\sum \alpha_t (x_t - \mu)^2 + \epsilon}$。
   - 拼接 $[\mu \,\|\, \sigma]$ 投影回 `in_dim`。
   - 输入张量：`(B, T, in_dim)`，输出：`(B, in_dim)`。

【验收单元测试要求】
- 编写 `tests/test_attention_blocks.py`：
  - 测试 `MultiHeadAttentivePooling` 在不同时间步长度（$T=10, 75, 300$）下输出维度始终为 `(B, in_dim)`；
  - 测试注意力权重 Softmax 在时间维度上的和严格等于 1.0；
  - 测试所有算子在 `loss.backward()` 下均能正常计算梯度。
```

---

### 提示词 2.2：统一 4 层 2D-CNN 骨干模型家族与工厂注册器
```markdown
【任务名称】构建统一 4 层 CNN 骨干与消融模型家族（M0 ~ M4）
【目标文件】`src/models/backbones.py`, `src/models/registry.py`

【技术规范与模型矩阵】
所有 CNN 骨干网络共享完全相同的基础卷积通道架构：
- Block 1: Conv2d(3 -> 32, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
- Block 2: Conv2d(32 -> 64, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
- Block 3: Conv2d(64 -> 128, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2)
- Block 4: Conv2d(128 -> 256, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 1)

注册以下对照模型：
1. `cnn_base` (M0 基线)：4 层 CNN + 末端全局平均池化 AdaptiveAvgPool2d(1) -> 线性分类头。
2. `cnn_se` (M1 通道)：4 个卷积块内部均插入 SELayer(reduction=16) + 末端全局平均池化。
3. `cnn_mhap` (M2 时域)：4 层纯 CNN 提取特征后，展平为 `(B, T, 256*F_reduced)`，投影至 256 维，末端接 4 头 MHAP 池化。
4. `cnn_se_mhap` (M3 复合)：同时在浅层开启 SELayer 并在末端开启 MHAP。
5. `w2v2_mhap` (M4 预训练底座)：冻结 wav2vec2-base 提取的 768 维帧级特征序列，接完全相同的 MHAP 头。

统一模型工厂接口：`get_model(name: str, n_classes: int = 8) -> nn.Module`。
```

---

## 阶段三：消融实验驱动器与 5 折基准评估 (Phase 3)

### 提示词 3.1：说话人无关 5 折训练引擎与评估监控
```markdown
【任务名称】构建支持断点续跑、SpecAugment 与 Macro-F1 监控的 5 折训练引擎
【目标文件】`src/engine/trainer.py`, `src/engine/metrics.py`, `scripts/run_ablation.py`

【技术规范与超参设定】
1. 训练超参数（严格受控，不可随意更改）：
   - 优化器：AdamW(lr=3e-4, weight_decay=1e-4)；
   - 调度器：CosineAnnealingLR(T_max=30, eta_min=1e-6)；
   - 批大小：batch_size=32；最大轮数：epochs=30；早停耐受：patience=8；
   - 梯度裁剪：clip_grad_norm = 1.0；
   - 随机种子：固定 seed=42。
2. 数据增强（仅在训练折生效，验证折严禁开启）：
   - SpecAugment：时间掩码（Time Masking, 最大 30 帧）与频率掩码（Freq Masking, 最大 16 个频带）。
3. 评估指标记录与持久化：
   - 每轮验证计算：Macro-F1 (主指标)、WAR (总体准确率)、UAR (宏平均召回率)；
   - 逐 Epoch 记录收敛轨迹保存至 `results/convergence_curves.csv`；
   - 逐折保存最佳模型权重 `outputs/{model}/fold{k}/best.pth` 与最终指标至 `results/fold_results.csv`。
4. 基准表格聚合：
   - 运行完成后自动计算跨 5 折的 mean ± std，格式化输出为 `results/benchmark.md`。
```

---

## 阶段四：WORLD 声码器物理因果闭环与交互探针 (Phase 4)

### 提示词 4.1：8 情绪声学物理画像库构建与参数域干预引擎
```markdown
【任务名称】提取 8 类情绪声学画像库并实现 WORLD 物理参数干预算子
【目标文件】`src/causal/vocoder.py`, `src/causal/modifier.py`, `scripts/extract_priors.py`

【技术规范与物理公式】
1. 情绪物理画像提取（`acoustic_priors.json`）：
   - 在训练集演员（Actor 1~18）上运行 WORLD 声码器（Harvest/CheapTrick/D4C），统计 8 类情绪的 4 个物理声学基准：
     - F0 均值 (Hz)
     - log-F0 标准差 (抑扬顿挫起伏度)
     - 能量 RMS (dB)
     - 语速时长分布
2. 物理规则干预算子：
   - 音高平移：$\Delta \text{semitones} = 12 \times \log_2(F_{0,\text{target}} / F_{0,\text{src}})$；
   - 音高方差缩放：$F_{0,\text{new}} = \mu_{\text{target}} \cdot (F_0 / \mu_{\text{src}})^{\sigma_{\text{target}} / \sigma_{\text{src}}}$；
   - 能量增益：$\Delta \text{dB} = \text{RMS}_{\text{target}} - \text{RMS}_{\text{src}}$（带 $\pm 10\text{dB}$ 自然性硬限幅）；
   - 时长伸缩：time_rate 帧长重采样。
3. 重建与合成：
   - 调用 `pyworld.synthesize(f0_mod, sp, ap, fs=16000)` 生成物理变声后的新音频。
```

---

### 提示词 4.2：8×8 情绪翻转率 (EFR) 自动化检验矩阵
```markdown
【任务名称】构建 WORLD 变声闭环回灌评估脚本与 8×8 EFR 矩阵生成器
【目标文件】`src/causal/efr_evaluator.py`, `scripts/run_causal_efr.py`

【技术规范与因果指标】
1. 测试数据源：仅使用未见说话人（Test Actor 21 ~ 24，共 240 条样本）。
2. 闭环评估流程：
   - 读取每条测试样本（名义源情绪 A）；
   - 依据声学画像库，分别生成转换为目标情绪 B（共 7 类）的变声音频；
   - 送入训练好的 `cnn_mhap` 识别模型进行分类预测；
   - 统计 $\text{EFR}(A \to B) = \frac{\text{被判为 } B \text{ 的样本数}}{\text{源情绪为 } A \text{ 的总样本数}}$。
3. 产出交付物：
   - 8×8 情绪翻转率热力图 `results/efr_matrix.png`；
   - 详细因果归因分析报告 `results/EVAL.md`（包含对角线 Sanity 准确率与典型翻转/失败对分析）。
```

---

### 提示词 4.3：Streamlit 交互式声学与注意力可视化面板
```markdown
【任务名称】构建可视化 Web 交互面板（声学谱图 + 多头注意力热力图 + 变声闭环）
【目标文件】`web_app/app.py`

【功能模块】
1. Tab 1：【语音情感感知与注意力透视 (Voice2Mood)】
   - 支持上传本地 wav 或选择示例音频；
   - 实时渲染 Log-Mel 谱图、预测概率柱状图；
   - 渲染 4 个 Attention Head 在时间轴上的注意力热力图（展示各 Head 关注的重音/语调区域）。
2. Tab 2：【WORLD 物理变声与因果回灌 (Mood2Voice)】
   - 选择目标情绪（Angry, Happy, Sad, Calm 等）与干预强度 Intensity (0.0 ~ 1.0)；
   - 实时播放变声合成后的音频，并显示修改后的 F0 音高轨迹；
   - 一键回灌给识别模型，实时展示情绪置信度翻转过程。
```

---

*手册状态：定稿 (v1.0)*  
*适用工作区：仓库根目录（`<repo>/`，运行脚本请从仓库根执行）*
