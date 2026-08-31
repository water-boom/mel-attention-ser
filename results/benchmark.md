# 5-Fold Speaker-Independent Benchmark Report

| Model Name | Macro-F1 (mean±std) | WAR / Accuracy | UAR / Recall | Mechanism |
|---|---|---|---|---|
| `cnn_base` | **0.4800 ± 0.0753** | 0.5372 ± 0.0693 | 0.5081 ± 0.0670 | 4-Layer CNN + GAP (Baseline) |
| `cnn_se` | **0.4400 ± 0.0413** | 0.5015 ± 0.0382 | 0.4708 ± 0.0370 | 4-Layer CNN + Channel SE Attention |
| `cnn_coord` | **0.4706 ± 0.0630** | 0.5305 ± 0.0593 | 0.4998 ± 0.0551 | 4-Layer CNN + Coordinate Spectro-Temporal Attention |
| `cnn_mhap` | **0.5578 ± 0.0503** | 0.5745 ± 0.0413 | 0.5658 ± 0.0449 | 4-Layer CNN + 4-Head Temporal Pooling (Ours) |
| `cnn_asp` | **0.5581 ± 0.0907** | 0.5743 ± 0.0813 | 0.5641 ± 0.0933 | 4-Layer CNN + Attentive Statistics Pooling (1st & 2nd Order) |
