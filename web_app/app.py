"""Streamlit Interactive Web Probe for Spectro-Temporal Attention & Causal Analysis.

Supports:
- Real-time microphone audio recording (st.audio_input) & WAV file upload
- 2D Log-Mel Spectrogram & Multi-Head Attention heatmaps
- Real-time causal frame masking (Top-K vs Bottom-K)
- WORLD vocoder acoustic physical manipulation & emotion flip verification
- 5-Fold Benchmark Report & Training Dynamics charts viewer
"""

import io
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn.functional as F

from src.data.frontend import AudioFrontend
from src.data.dataset import load_audio_file
from src.models.registry import build_model, list_models
from src.causal.vocoder import AcousticModifier, WorldVocoder, PYWORLD_AVAILABLE
from src.causal.masking_probe import apply_frame_mask

EMOTION_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOTION_EMOJIS = ["😐", "😌", "😄", "😢", "😡", "😨", "🤢", "😲"]

st.set_page_config(
    page_title="Spectro-Temporal Attention SER Lab",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource
def get_frontend():
    return AudioFrontend(sample_rate=16000, n_mels=128, target_frames=300)


@st.cache_resource
def load_ser_model(model_name: str = "cnn_mhap"):
    model = build_model(model_name, num_classes=8)
    ckpt_path = f"results/checkpoints/{model_name}_best.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data
def load_norm_stats():
    stats_path = "results/checkpoints/norm_stats.pt"
    if os.path.exists(stats_path):
        return torch.load(stats_path, map_location="cpu")
    return None


@st.cache_data
def load_priors():
    path = "configs/acoustic_priors.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    st.title("🎙️ 时频注意力机制实验室 (Spectro-Temporal Attention Lab)")
    st.markdown(
        "> **探究梅尔谱物理本质与注意力机制的真实作用机理** | 实时录音探针 · 5折消融报告 · 动力学演进 · 因果干预闭环"
    )

    frontend = get_frontend()
    norm_stats = load_norm_stats()
    priors = load_priors()

    tab1, tab2, tab3 = st.tabs([
        "🎙️ Tab 1: 实时录音与注意力探针",
        "🎛️ Tab 2: 声学物理干预与 EFR 变声",
        "📈 Tab 3: 5-Fold 消融报告与动力学图表",
    ])

    # -------------------------------------------------------------
    # TAB 1: Real-Time Audio Recording & Attention Probing
    # -------------------------------------------------------------
    with tab1:
        st.subheader("1. 音频输入与时频注意力分析")
        col_ctrl1, col_ctrl2 = st.columns([1, 1])

        with col_ctrl1:
            input_mode = st.radio(
                "选择音频输入方式",
                ["🎙️ 实时麦克风录音", "🎧 内置 8 类标准情绪测试样本", "📁 上传本地 WAV 文件"],
                horizontal=True,
            )

        with col_ctrl2:
            selected_model_name = st.selectbox(
                "选择模型架构 (Attention Zoo)",
                list_models(),
                index=list_models().index("cnn_mhap") if "cnn_mhap" in list_models() else 0,
            )
            model = load_ser_model(selected_model_name)
            ckpt_file = f"results/checkpoints/{selected_model_name}_best.pt"
            if os.path.exists(ckpt_file):
                st.caption(f"✅ 已加载 5-Fold 训练最优权重: `{ckpt_file}`")
            else:
                st.caption("ℹ️ 当前使用模型权重")

        wav = None

        if input_mode == "🎙️ 实时麦克风录音":
            st.markdown("##### 🎤 点击下方录音按钮说话（建议尝试用平缓或激动的语气说话）：")
            audio_data = st.audio_input("录制你的语音")
            if audio_data is not None:
                wav_bytes = audio_data.read()
                wav, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                if sr != 16000:
                    import scipy.signal
                    wav = scipy.signal.resample_poly(wav, 16000, sr).astype(np.float32)

        elif input_mode == "🎧 内置 8 类标准情绪测试样本":
            demo_emotion = st.selectbox(
                "选择测试样本的真实情感标签",
                list(range(8)),
                format_func=lambda x: f"{EMOTION_EMOJIS[x]} 真实标签: {EMOTION_NAMES[x].upper()} ({priors.get(str(x), {}).get('description', '')})",
                index=0,
            )
            # Find matching sample from test data
            data_dir = "D:/learn/hdu_class/AI_Introduction/Mood2Voice/speech-emotion-recognition-ravdess-data/speech-emotion-recognition-ravdess-data"
            import glob
            sample_candidates = glob.glob(os.path.join(data_dir, f"**/*-0{demo_emotion+1:01d}-*.wav"), recursive=True)
            if sample_candidates:
                sample_path = sample_candidates[0]
                wav = load_audio_file(sample_path, 16000).numpy()
                st.caption(f"已加载标准测试样本: `{os.path.basename(sample_path)}`")
            else:
                t_demo = np.linspace(0, 3.0, 48000, endpoint=False)
                wav = (0.3 * np.sin(2 * np.pi * 220 * t_demo)).astype(np.float32)

        elif input_mode == "📁 上传本地 WAV 文件":
            uploaded_file = st.file_uploader("上传一段 16kHz WAV 语音", type=["wav"])
            if uploaded_file is not None:
                wav_bytes = uploaded_file.read()
                wav, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                if wav.ndim > 1:
                    wav = np.mean(wav, axis=1)
                if sr != 16000:
                    import scipy.signal
                    wav = scipy.signal.resample_poly(wav, 16000, sr).astype(np.float32)

        if wav is not None:
            st.audio(wav, sample_rate=16000)
            wav_tensor = torch.from_numpy(wav).float()

            # Feature Extraction
            with torch.no_grad():
                feat = frontend(wav_tensor)  # (1, 3, 128, 300)
                # In-fold standardization if stats exist
                if norm_stats is not None:
                    feat_norm = (feat - norm_stats["mean"]) / (norm_stats["std"] + 1e-6)
                else:
                    feat_norm = feat

                logits = model(feat_norm)
                probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
                attn_maps = model.get_attention_maps(feat_norm)  # (1, K, T) or None

            # Display Mel-spectrogram & Attention Maps
            col_spec, col_pred = st.columns([3, 2])

            with col_spec:
                st.markdown("#### 2D Log-Mel 时频谱图 (128 Bins × 300 Frames)")
                log_mel = feat[0, 0].cpu().numpy()
                fig, ax = plt.subplots(figsize=(8, 3.5))
                im = ax.imshow(log_mel, origin="lower", aspect="auto", cmap="viridis")
                ax.set_title("Log-Mel Spectrogram Energy")
                ax.set_xlabel("Time Frames (~4.8s)")
                ax.set_ylabel("Mel Frequency Bins (0~8000 Hz)")
                fig.colorbar(im, ax=ax)
                st.pyplot(fig)
                plt.close()

                if attn_maps is not None:
                    st.markdown("#### 多头注意力时域激活热力图 (Attention Weights α(t))")
                    attn_np = attn_maps[0].cpu().numpy()  # (K, T_sub)
                    fig_attn, ax_attn = plt.subplots(figsize=(8, 2.5))
                    for h_idx in range(attn_np.shape[0]):
                        ax_attn.plot(attn_np[h_idx], label=f"Head {h_idx+1}", lw=1.8)
                    ax_attn.set_title("Attention Weight Distribution Across Time")
                    ax_attn.set_xlabel("Time Frame (t)")
                    ax_attn.set_ylabel("Weight Alpha(t)")
                    ax_attn.grid(True, linestyle="--", alpha=0.5)
                    ax_attn.legend()
                    st.pyplot(fig_attn)
                    plt.close()

            with col_pred:
                st.markdown("#### 情绪分类预测分布")
                pred_idx = int(np.argmax(probs))
                st.success(f"**预测最高类**: {EMOTION_EMOJIS[pred_idx]} **{EMOTION_NAMES[pred_idx].upper()}** ({probs[pred_idx]*100:.1f}%)")

                for idx, (name, emoji, p) in enumerate(zip(EMOTION_NAMES, EMOTION_EMOJIS, probs)):
                    st.progress(float(p), text=f"{emoji} {name.capitalize()}: {p*100:.1f}%")

                st.markdown("---")
                st.markdown("#### 🧪 实时因果遮蔽探针 (Causal Masking)")
                mask_ratio = st.slider("遮蔽比例 (Mask Ratio)", 0.0, 0.6, 0.0, 0.05)
                mask_mode = st.radio("遮蔽模式", ["Top-K (遮蔽关键帧)", "Bottom-K (遮蔽低权帧)"], horizontal=True)

                if mask_ratio > 0.0:
                    mode_str = "top" if "Top-K" in mask_mode else "bottom"
                    if attn_maps is not None:
                        scores_sub = attn_maps.mean(dim=1, keepdim=True)
                        scores = F.interpolate(scores_sub, size=feat_norm.shape[-1], mode="nearest").squeeze(1)
                    else:
                        scores = feat_norm[:, 0].mean(dim=1)
                    masked_feat = apply_frame_mask(feat_norm, scores, mask_ratio, mode=mode_str)
                    with torch.no_grad():
                        masked_logits = model(masked_feat)
                        masked_probs = F.softmax(masked_logits, dim=-1).squeeze(0).numpy()
                    m_pred_idx = int(np.argmax(masked_probs))
                    st.warning(f"**遮蔽后预测**: {EMOTION_EMOJIS[m_pred_idx]} **{EMOTION_NAMES[m_pred_idx].upper()}** ({masked_probs[m_pred_idx]*100:.1f}%)")

    # -------------------------------------------------------------
    # TAB 2: Causal Acoustic Modification & Feedback
    # -------------------------------------------------------------
    with tab2:
        st.subheader("2. WORLD 声码器物理干预与 EFR 翻转率闭环")
        st.markdown(
            "通过显式修改语音的**物理基频 ($F_0$)、能量强度 (RMS) 与语速时长**，送回分类器检验预测结果是否发生符合物理声学规律的定向翻转。"
        )

        col_ctrl, col_synth = st.columns([1, 1])

        with col_ctrl:
            target_emotion = st.selectbox(
                "目标情绪先验 (Target Emotion Prior)",
                list(range(8)),
                format_func=lambda x: f"{EMOTION_EMOJIS[x]} {EMOTION_NAMES[x].capitalize()} ({priors.get(str(x), {}).get('description', '')})",
                index=4,  # default Angry
            )

            intensity = st.slider("干预强度 (Intervention Intensity)", 0.0, 1.5, 1.0, 0.1)

            st.markdown("##### 显式声学物理微调参数：")
            tgt_info = priors.get(str(target_emotion), {})
            st.write(f"- **目标基频基准**: {tgt_info.get('f0_mean_hz', 200.0)} Hz")
            st.write(f"- **目标能量强度**: {tgt_info.get('rms_db', -35.0)} dB")
            st.write(f"- **目标起伏方差**: {tgt_info.get('log_f0_std', 0.25)}")

            delta_semi = st.slider("音高平移 (Delta Semitones)", -12.0, 12.0, 4.0, 0.5)
            gain_db = st.slider("能量增益 (Gain dB)", -15.0, 15.0, 6.0, 0.5)

        with col_synth:
            st.markdown("##### 物理合成与回灌校验")
            if st.button("🚀 执行声码器物理干预并重合成", type="primary"):
                if wav is None:
                    st.warning("请先在 Tab 1 中录音或上传音频。")
                else:
                    with st.spinner("正在通过声码器修改声学物理参数..."):
                        if PYWORLD_AVAILABLE:
                            vocoder = WorldVocoder(sample_rate=16000)
                            try:
                                mod_wav = AcousticModifier.modify_speech(
                                    wav,
                                    vocoder,
                                    delta_semitones=delta_semi,
                                    gain_db=gain_db,
                                    intensity=intensity,
                                    f0_var_scale=float(tgt_info.get("log_f0_std", 0.25)) / 0.2,
                                )
                                st.audio(mod_wav, sample_rate=16000)

                                # Re-feed to classifier
                                mod_wav_t = torch.from_numpy(mod_wav).float()
                                with torch.no_grad():
                                    mod_feat = frontend(mod_wav_t)
                                    if norm_stats is not None:
                                        mod_feat = (mod_feat - norm_stats["mean"]) / (norm_stats["std"] + 1e-6)
                                    mod_logits = model(mod_feat)
                                    mod_probs = F.softmax(mod_logits, dim=-1).squeeze(0).numpy()
                                    mod_pred = int(np.argmax(mod_probs))

                                st.success(
                                    f"🎉 **回灌分类器预测结果**: {EMOTION_EMOJIS[mod_pred]} "
                                    f"**{EMOTION_NAMES[mod_pred].upper()}** ({mod_probs[mod_pred]*100:.1f}%)"
                                )
                            except Exception as e:
                                st.error(f"合成过程发生异常: {e}")
                        else:
                            st.warning("系统未安装 pyworld，无法在当前环境执行声码器重合成。")

    # -------------------------------------------------------------
    # TAB 3: Benchmark Report & Dynamics Charts
    # -------------------------------------------------------------
    with tab3:
        st.subheader("3. 5-Fold 说话人无关基准报告与动力学分析")

        report_path = "results/benchmark.md"
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("尚未生成 benchmark.md，请先运行 `python scripts/run_5fold.py`。")

        st.markdown("---")
        st.markdown("#### 📈 实验分析图表展示")
        col_fig1, col_fig2 = st.columns(2)

        fig_entropy = "results/figures/dynamics_entropy.png"
        fig_diversity = "results/figures/head_diversity.png"
        fig_causal = "results/figures/causal_masking_curve.png"

        with col_fig1:
            if os.path.exists(fig_entropy):
                st.image(fig_entropy, caption="图 1：注意力信息熵演进轨迹 (Entropy Evolution)")
            if os.path.exists(fig_causal):
                st.image(fig_causal, caption="图 3：Top-K vs Bottom-K 因果遮蔽衰减曲线")

        with col_fig2:
            if os.path.exists(fig_diversity):
                st.image(fig_diversity, caption="图 2：多头正交分工度演进 (Head Diversity)")


if __name__ == "__main__":
    main()
