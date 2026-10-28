# Submission Manifest

本文件说明课程提交包的内容、来源边界和核验方式。

---

## 1. 提交内容

| 文件/目录 | 作用 |
|---|---|
| `README.md` | 提交入口：复现范围、主要结果、运行方法 |
| `CHECKPOINTS.md` | 两个推理 checkpoint 的来源、大小、SHA256、放置路径 |
| `SUBMISSION.md` | 本文件：文件清单与来源边界 |
| `README_ORIGINAL.md` | 原作者 README，保留 |
| `LICENSE` | 原始 MIT 许可证，Copyright (c) 2022 ScienceNLP-Lab |
| `requirements-repro.txt` | 实际验证过的 Python 环境 |
| `requirements.txt` | 原作者依赖，保留但未在本机验证 |
| `train_configs.json` | 推理实际读取的配置 |
| `Data/` | 作者发布的数据；`annotations.zip` 含 citation-level gold |
| `CitationAccuracyClassificationCode/multivers/` | 原作者 MultiVerS 代码 |
| `CitationAccuracyClassificationCode/train_target.py` | 原作者训练入口 |
| `CitationAccuracyClassificationCode/prepare_data.py` | 细标签折叠为 3 类，生成 `data_train/` |
| `CitationAccuracyClassificationCode/bm25t5.ipynb` | 原论文 BM25 + MonoT5 检索流程 |
| `CitationAccuracyClassificationCode/run_predict_testonly.py` | 本复现的推理入口（只构建 test fold） |
| `CitationAccuracyClassificationCode/eval_predictions.py` | 本复现的评测脚本（per-class P/R/F1、混淆矩阵） |
| `CitationAccuracyClassificationCode/data_train/target/scifact_10/` | 已生成的推理/评测数据 |
| `CitationAccuracyClassificationCode/predictions/` | 冻结的预测、指标、bootstrap、错误分析、diff |
| `CitationAccuracyClassificationCode/REPORT_OFFICIAL_CKPT.md` | 主报告 |
| `CitationAccuracyClassificationCode/REPRODUCE.md` | 完整复现步骤 |
| `CitationAccuracyClassificationCode/checkpoints/fast_download.sh` | 可选下载脚本，不含权重 |

---

## 2. 原始作者的工作 vs. 本复现新增的工作

### 原作者提供

- 论文、数据、`Data/annotations.zip`；
- `multivers/` 下的 MultiVerS 代码；
- `train_target.py`、`bm25t5.ipynb`、`get_checkpoint.py`；
- 发布的 `bestModel.ckpt`；
- MIT `LICENSE` 和原始 README。

### 本复现新增

- `prepare_data.py`：把细粒度标签折叠为论文的 3 类，生成 `data_train/`；
- `run_predict_testonly.py`：推理时只构建 test fold，跳过无用的 train/dev 预处理；
- `eval_predictions.py`：实现 citation-level / pair-level 评测、per-class P/R/F1、混淆矩阵；
- `predictions/`：冻结的逐 pair 预测、评测 JSON、bootstrap CI、逐样本错误分析；
- `REPORT_OFFICIAL_CKPT.md`：结果、混淆矩阵、8 个错误案例、原因分析、局限性；
- `REPRODUCE.md`：从环境到推理/评测的完整命令；
- `README.md`、`CHECKPOINTS.md`、`SUBMISSION.md`：提交说明。

### 对原作者代码的修改

仅运行兼容性修改，不改变模型结构、权重加载或预测逻辑：

- `multivers/model.py`：注释掉未使用的 `bertviz`、`focal_loss` 导入；
- `multivers/data_train.py`：删除调试 `print`；标签重加权改为 `reindex(..., fill_value=0).clip(lower=1)`，避免 debug 子集缺少某类时报 `KeyError`；
- 新增 `run_predict_testonly.py` 和 `eval_predictions.py`。

完整 diff：`CitationAccuracyClassificationCode/predictions/repro_working_tree.diff`。

---

## 3. 复现边界

> **本提交没有重新训练模型。**  
> 所有结果来自作者发布的 `bestModel.ckpt` 的推理。

具体边界：

- 使用作者 `bestModel.ckpt`（epoch 5，global_step 1535）；
- 在作者发布的 citation–摘要段落 pair 数据上推理；
- 测试集 606 claims / 41,607 pairs；
- citation-level 可评测 605 条（id 106 的引用文本对应两个冲突细标签，按协议剔除）；
- 论文 Table 4 使用 top-20 BM25+MonoT5 句子候选，该数据未发布，因此本工作不是严格 Table 4 复现；
- 论文 micro-F1 0.59、macro-F1 0.52 均落在本实验 bootstrap 95% CI 内，但不能据此声称严格复现。

---

## 4. 核验清单

老师/助教可按以下顺序核验：

```text
[ ] 1. 读 README.md 的“复现边界”和“主要结果”
[ ] 2. 打开 predictions/eval_official.json 对照指标
[ ] 3. 不装 GPU 环境，直接运行 eval_predictions.py 复算 citation-level 指标
[ ] 4. 打开 predictions/error_analysis.json 查看 605 条逐样本结果和 8 个案例
[ ] 5. 打开 predictions/repro_working_tree.diff 确认没有改模型/预测逻辑
[ ] 6. 按 CHECKPOINTS.md 下载两个 checkpoint，按 README 第 4 节重跑推理
```

最小复算命令：

```bash
cd CitationAccuracyClassificationCode

python eval_predictions.py \
  --predictions predictions/test_predictions_official.jsonl \
  --claims data_train/target/scifact_10/claims_test.jsonl \
  --annotations ../Data/annotations.zip \
  --output predictions/eval_check.json
```

预期主要指标：

```text
citation-level micro-F1 = 0.6182
citation-level macro-F1 = 0.4914
```

---

## 5. 未提交的内容

| 未提交 | 原因 |
|---|---|
| `checkpoints_official/bestModel.ckpt` | 4.9GB，GitHub 单文件限制 100MB；用 `CHECKPOINTS.md` 链接 |
| `checkpoints/healthver.ckpt` | 4.9GB，仅训练需要 |
| `checkpoints/longformer_large_science.ckpt` | 1.5GB，推理需要，但用链接分发 |
| `.git/` | 原作者仓库历史，348MB，与复现无关 |
| `__pycache__/`、`*.pyc` | 缓存 |
| `lightning_logs/`、`output.txt`、`checkpoints_user/` | 训练日志/中间产物 |
| `.venv/`、conda 环境 | 用 `requirements-repro.txt` 替代 |

---

## 6. 引用

- 论文：Sarol et al. *Assessing Citation Integrity in Biomedical Publications: Annotation and NLP Models.* Bioinformatics, 2024.
- 原始代码：https://github.com/ScienceNLP-Lab/Citation-Integrity
- MultiVerS：https://github.com/dwadden/multivers

原始代码使用 MIT 许可证；本提交保留原始 `LICENSE` 与版权声明。
