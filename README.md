# Citation Integrity Reproduction

本仓库复现论文 **“Assessing Citation Integrity in Biomedical Publications: Annotation and NLP Models”**（Sarol et al., Bioinformatics 2024）中的 citation accuracy 分类实验。

> **复现边界：本提交是基于作者发布的 `bestModel.ckpt` 进行推理、评测和错误分析，没有重新训练模型。**
>
> 原始论文的 Table 4 使用 BM25 + MonoT5 检索出的 top-20 句子作为候选；作者发布的数据是 citation–候选摘要段落 pair。两者数据口径不同，因此本工作不是严格的 Table 4 复现，而是“作者 checkpoint + 公开发布数据”的推理复现。

---

## 1. 主要结果

测试集：606 条 claim，41,607 个 claim–paragraph pair。  
Citation-level 评测：605 条（id 106 的引用文本对应两个冲突的细标签，按协议剔除）。

| 指标 | 数值 |
|---|---:|
| Citation-level micro-F1 | **0.6182** |
| Citation-level macro-F1 | **0.4914** |
| Micro-F1 95% bootstrap CI | [0.5785, 0.6579] |
| Macro-F1 95% bootstrap CI | [0.4338, 0.5453] |
| 错误数 | 231 / 605（38.2%） |

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ACCURATE | 0.7113 | 0.7850 | 0.7463 | 386 |
| NOT_ACCURATE | 0.3819 | 0.3254 | 0.3514 | 169 |
| IRRELEVANT | 0.4571 | 0.3200 | 0.3765 | 50 |

混淆矩阵（行 = gold，列 = prediction）：

| gold \ pred | ACCURATE | NOT_ACCURATE | IRRELEVANT |
|---|---:|---:|---:|
| ACCURATE | 303 | 73 | 10 |
| NOT_ACCURATE | 105 | 55 | 9 |
| IRRELEVANT | 18 | 16 | 16 |

Pair-level：micro-F1 0.8116，macro-F1 0.3488。该数值受类别极不平衡影响（98.15% 的 pair 为 IRRELEVANT），不能直接与 citation-level 指标比较。

论文报告的 micro-F1 0.59、macro-F1 0.52 均落在本实验的 bootstrap 95% CI 内，但数据口径不同，不能据此声称严格复现了 Table 4。

详细分析见：

- `CitationAccuracyClassificationCode/REPORT_OFFICIAL_CKPT.md`
- `CitationAccuracyClassificationCode/predictions/eval_official.json`
- `CitationAccuracyClassificationCode/predictions/bootstrap_ci.json`
- `CitationAccuracyClassificationCode/predictions/error_analysis.json`

---

## 2. 目录结构

```text
citation-integrity-repro/
├── README.md                       # 本文件
├── CHECKPOINTS.md                  # checkpoint 来源、大小、SHA256、放置路径
├── SUBMISSION.md                   # 提交清单与“原作者/我的工作”边界
├── README_ORIGINAL.md              # 原作者 README，保留
├── LICENSE                         # MIT (c) 2022 ScienceNLP-Lab
├── requirements.txt                # 原作者依赖，未在本机验证
├── requirements-repro.txt          # 实际验证过的环境
├── train_configs.json              # 推理时实际读取的配置
├── Data/                           # 作者发布数据（含 annotations.zip）
└── CitationAccuracyClassificationCode/
    ├── multivers/                  # 原作者代码
    ├── train_target.py             # 训练入口
    ├── prepare_data.py             # 细标签 -> 3 类，生成 data_train/
    ├── bm25t5.ipynb                # 原论文检索流程
    ├── get_checkpoint.py           # 下载基础 checkpoint
    ├── patch_mirror_compat.py      # 可选：镜像兼容
    ├── run_predict_testonly.py     # 本复现的推理入口
    ├── eval_predictions.py         # 本复现的评测脚本
    ├── data_train/target/scifact_10/
    ├── predictions/
    ├── REPORT_OFFICIAL_CKPT.md
    ├── REPRODUCE.md
    └── checkpoints/
        └── fast_download.sh        # 仅脚本，不含 .ckpt
```

推理需要把两个 checkpoint 放到：

```text
CitationAccuracyClassificationCode/checkpoints_official/bestModel.ckpt
CitationAccuracyClassificationCode/checkpoints/longformer_large_science.ckpt
```

`healthver.ckpt` 只在训练时需要，推理不需要。详见 `CHECKPOINTS.md`。

---

## 3. 不跑 GPU 也能复算指标

`eval_predictions.py` 只依赖 Python 标准库，不需要 PyTorch/GPU。用冻结的逐 pair 预测和 `Data/annotations.zip` 即可复算全部 citation-level 指标：

```bash
cd CitationAccuracyClassificationCode

python eval_predictions.py \
  --predictions predictions/test_predictions_official.jsonl \
  --claims data_train/target/scifact_10/claims_test.jsonl \
  --annotations ../Data/annotations.zip \
  --output predictions/eval_check.json
```

然后用 `predictions/eval_official.json` 对照。`eval_check.json` 是临时输出，可删除。

评测口径：

1. 每条 citation 有多个候选段落；
2. 选择 `P(NOT_ACCURATE) + P(ACCURATE)` 最大的候选；
3. 用该候选的 argmax 类别作为 citation 预测；
4. gold 来自 `Data/annotations.zip`，细标签按论文第 2.5.3 节折叠为 3 类：
   - `ACCURATE / INDIRECT / INDIRECT_NOT_REVIEW` → `ACCURATE`
   - `CONTRADICT / NOT_SUBSTANTIATE / OVERSIMPLIFY / MISQUOTE / ETIQUETTE` → `NOT_ACCURATE`
   - `IRRELEVANT` → `IRRELEVANT`

---

## 4. 重跑推理（需要 GPU 和 checkpoint）

环境：Python 3.9.25、PyTorch 1.8.1+cu111、transformers 4.2.2、pytorch-lightning 1.2.1、CUDA 11.1。  
实测：单张 RTX 3090 24GB，峰值约 19.3GB，耗时约 53 分钟。

```bash
cd CitationAccuracyClassificationCode

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python run_predict_testonly.py \
  --checkpoint_path checkpoints_official/bestModel.ckpt \
  --input_file data_train/target/scifact_10/claims_test.jsonl \
  --corpus_file data_train/target/scifact_10/corpus.jsonl \
  --output_file predictions/test_predictions_official.jsonl \
  --device 0 --num_workers 4
```

注意：

- 数据模块从仓库根目录的 `../train_configs.json` 读取配置，`eval_batch_size=2`；
- 命令行 `--batch_size` 不会改变数据模块的 batch size；
- `--corpus_file` 在当前代码路径中未被实际使用；
- `run_predict_testonly.py` 只跳过推理不需要的 train/dev 预处理，测试集构造与 `multivers/predict.py` 一致，不改变预测结果；
- 输出文件约 4.4MB，606 行。

---

## 5. 代码修改说明

相对原作者代码，改动仅为运行兼容性，不涉及模型结构、权重加载逻辑或预测逻辑：

| 文件 | 修改 |
|---|---|
| `multivers/model.py` | 注释掉未使用的 `bertviz`、`focal_loss` 导入 |
| `multivers/data_train.py` | 删除调试 `print`；标签重加权改用 `reindex(..., fill_value=0).clip(lower=1)`，避免 debug 子集缺少某类时报 `KeyError` |
| `run_predict_testonly.py` | 新增；仅构建 test fold，跳过 train/dev 分词 |
| `eval_predictions.py` | 新增；实现论文第 2.6 节评测与混淆矩阵 |

完整 diff 见 `CitationAccuracyClassificationCode/predictions/repro_working_tree.diff`。

---

## 6. 数据说明

- `Data/`：作者发布的数据，保持原样；`Data/annotations.zip` 用于 citation-level gold。
- `data_train/target/scifact_10/`：由 `prepare_data.py` 从 `Data/multivers-format/` 生成，已随包提供，可直接推理/评测。
- 测试集：606 claims，41,607 pairs，8,515 篇 corpus 文档。
- Citation-level gold：ACCURATE 386、NOT_ACCURATE 169、IRRELEVANT 50，共 605 条可评测。

重新生成 `data_train/`：

```bash
cd CitationAccuracyClassificationCode
python prepare_data.py
```

---

## 7. 复现结论与限制

1. **不是重新训练**：使用的是作者发布的 `bestModel.ckpt`（epoch 5，global_step 1535）。
2. **不是严格 Table 4**：作者未发布 top-20 BM25+MonoT5 检索数据；本实验在公开的 citation–摘要段落 pair 上评测。
3. **指标与论文统计上一致**：0.6182 / 0.4914 的 bootstrap CI 覆盖论文的 0.59 / 0.52。
4. **主要错误来源**：NOT_ACCURATE → ACCURATE（105 例）、ACCURATE → NOT_ACCURATE（73 例）；细标签中 CONTRADICT 0/12、MISQUOTE 1/9、OVERSIMPLIFY 16/47 正确，说明模型对语义方向、省略/限定条件不敏感。
5. **评测协议影响**：citation-level 聚合规则、候选段落选择、ETIQUETTE/IRRELEVANT 无证据段落，都会影响最终数字；这些均在报告中逐项说明。

---

## 8. 引用与许可

原始论文：

> Sarol et al. Assessing Citation Integrity in Biomedical Publications: Annotation and NLP Models. Bioinformatics, 2024.

原始代码：

> https://github.com/ScienceNLP-Lab/Citation-Integrity

本仓库保留原始 MIT `LICENSE`（Copyright (c) 2022 ScienceNLP-Lab）。  
原作者 README 保留为 `README_ORIGINAL.md`。本复现新增的数据转换、推理入口、评测脚本、bootstrap、错误分析与报告，见 `SUBMISSION.md`。
