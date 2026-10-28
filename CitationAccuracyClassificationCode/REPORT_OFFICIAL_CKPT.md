# 作者 checkpoint 推理复现与错误分析

**论文**：Sarol et al., *Assessing citation integrity in biomedical publications: corpus annotation and NLP models*, Bioinformatics 40(7), 2024, btae420.

**代码目录**：`/home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode`

**日期**：2026-10-28

**一句话结论**：在作者发布的 `bestModel.ckpt` 上做纯推理，citation-level micro-F1 = **0.6182**、macro-F1 = **0.4914**；论文报告值为 0.59 / 0.52，两个值都落在 95% bootstrap 置信区间内。聚合指标与论文一致，但错误集中在 NOT_ACCURATE 的细分类型（CONTRADICT / OVERSIMPLIFY / ETIQUETTE）和 IRRELEVANT 上。

---

## 0. 实验性质声明（必读）

> **本报告是“基于作者 checkpoint 的推理复现”（inference-only reproduction），不是训练复现。**
>
> - 使用的是作者公开在 Google Drive 上发布的 `bestModel.ckpt`，不是本次实验训练出来的权重。
> - 本次实验**没有更新任何模型参数**，没有运行反向传播，没有用训练集/验证集重新训练或微调。
> - 推理前后 checkpoint 文件只被读取，未被修改；文件 SHA256 见 §1.2。
> - 因此，下面的数字反映的是“作者权重 + 发布数据 + 本报告重建的评测管线”的一致性，不包含训练随机性带来的方差。

---

## 1. 固定当前实验结果

### 1.1 实验对象与边界

| 项目 | 内容 |
|---|---|
| 任务 | 引用准确性三分类：ACCURATE / NOT_ACCURATE / IRRELEVANT |
| 模型 | MultiVerS + Longformer-large-4096（作者发布的 citation-accuracy checkpoint） |
| 实验类型 | **推理复现，无重新训练** |
| 推理数据 | `data_train/target/scifact_10/claims_test.jsonl` + `corpus.jsonl` |
| 评测级别 | citation level（主指标）+ claim–candidate abstract pair level（参考） |
| 是否使用作者 checkpoint | **是** |
| 是否重新训练 | **否** |

**术语说明**：仓库和论文中常把候选单位称为 paragraph；按发布数据的实际结构，每个候选是
`corpus.jsonl` 中的一篇 reference 的 `title + abstract`（abstract 为句子列表），模型把
title 和句子用分隔符拼接后作为候选文本。下文的“候选文档 / 证据摘要”均指这个单位。
`predictions/error_analysis.json` 中的 `chosen_text` 是候选文本的前 800 字符，长摘要会截断。

### 1.2 Checkpoint 来源

| 项目 | 值 |
|---|---|
| 本地路径 | `checkpoints_official/bestModel.ckpt` |
| 来源 | `https://drive.google.com/drive/folders/11b6Z8iv2FXObWmLaqfYzgUQsaL4QgTT2` |
| Google Drive file id | `1fnNiY_ECuOgDt561S6pmc8hzI4rnjn5Y` |
| 下载方式 | `gdown 4.5.4`，经本机 HTTP 代理 `127.0.0.1:17900` |
| 文件大小 | `5,253,790,961` bytes（约 4.9 GiB） |
| SHA256 | `b77f7afc6eeba2d137189b1c91d3c0baf51b8966af93cfd0326777207fc44bf4` |
| checkpoint 内部 epoch | 5 |
| checkpoint 内部 global_step | 1535 |
| `num_training_instances` | 2454 |
| `datasets` | `scifact_10` |
| `starting_checkpoint` | `checkpoints/healthver.ckpt` |
| `encoder_name`（hparams 记录值） | `longformer-large-science` |
| `lr` | `1e-5` |
| `train_batch_size` / `eval_batch_size` | 1 / 2 |
| `accumulate_grad_batches`（训练配置） | 8 |
| `num_labels` | 3 |
| callback monitor | `valid_sentence_label_f1` |
| callback best score | 0.1265 |
| callback best path | `.../epoch=1-step=613.ckpt`（发布文件本身是 epoch 5） |

**Encoder 加载路径**：`multivers/model.py:_get_encoder()` 先加载
`allenai/longformer-large-4096`，再叠加 Longformer science checkpoint 权重。
checkpoint 里的 `encoder_name=longformer-large-science` 是训练时记录的标签，实际的
初始化路径以 `model.py` 为准。

**训练设定的重要证据**：`num_training_instances=2454`、`global_step=1535`、epoch 5，
配合 `train_batch_size=1`、`accumulate_grad_batches=8`，可以反推出
`2454 / (1 × 8) ≈ 307` step/epoch，`307 × 5 = 1535` step。这说明作者训练时的实例数
在 **2.4k 量级**，量级上对应**每个 citation 约一个训练样本**，而不是发布数据里的
197,418 个 claim–abstract pair。该点对“能否严格对齐论文 Table 4”很关键，见 §1.12。

### 1.3 数据版本

数据由仓库的 `prepare_data.py` 从 `Data/multivers-format` 生成，并把论文的细粒度标签
折叠为 3 类：

- ACCURATE：`ACCURATE / INDIRECT / INDIRECT_NOT_REVIEW`
- NOT_ACCURATE：`CONTRADICT / NOT_SUBSTANTIATE / OVERSIMPLIFY / MISQUOTE / ETIQUETTE`
- IRRELEVANT：`IRRELEVANT`

| Split | Claims | Claim–candidate pairs |
|---|---:|---:|
| train | 2,141 | 197,418 |
| dev | 316 | 24,797 |
| test | 606 | 41,607 |
| corpus docs | — | 8,515 |

**Test citation-level gold（可评 605 条）**：

| 类别 | 数量 |
|---|---:|
| ACCURATE | 386 |
| NOT_ACCURATE | 169 |
| IRRELEVANT | 50 |
| 合计 | 605 |

**数据文件 SHA256**：

| 文件 | SHA256 |
|---|---|
| `claims_train.jsonl` | `062825326986d1724ab52ea311614cc4ffbf0efbd0c91c7d22a70d1e80720be4` |
| `claims_dev.jsonl` | `8581edfa869f2695223cb9a9b017a0a277bb26c1a581f8b520fa3884c32d512a` |
| `claims_test.jsonl` | `8ba5040586a86a20a7f6c1673e77cd5132f908509f45d0d966c09565ccb42b6d` |
| `corpus.jsonl` | `bae9598556411c3d855877bea4063d09f01167ce6d8c9d2c5912f880fa77c92f` |
| `../Data/annotations.zip` | `cf22dea33606f355c8d9684dd8d18234be559bf40a45968fd2829cdf66e5797e` |

仓库 commit：`e9823957bb263db9ae402351b76d88ff1a723712`。工作区改动见
`predictions/repro_working_tree.diff`；改动只删除调试打印/未使用 import、并让标签
reindex 更稳健，不改变推理预测。

### 1.4 运行环境

| 组件 | 版本 |
|---|---|
| Python | 3.9.25 |
| PyTorch | 1.8.1+cu111 |
| transformers | 4.2.2 |
| pytorch-lightning | 1.2.1 |
| tokenizers | 0.9.4 |
| CUDA | 11.1 |
| GPU | NVIDIA GeForce RTX 3090（24 GB） |
| OS | Linux 5.15.0-138-generic x86_64, glibc 2.35 |

### 1.5 推理命令

```bash
cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode
source /home/chenjw/anaconda3/etc/profile.d/conda.sh
conda activate citation-integrity

CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=false \
python run_predict_testonly.py \
  --checkpoint_path checkpoints_official/bestModel.ckpt \
  --input_file data_train/target/scifact_10/claims_test.jsonl \
  --corpus_file data_train/target/scifact_10/corpus.jsonl \
  --output_file predictions/test_predictions_official.jsonl \
  --device 0 \
  --num_workers 4
```

- `run_predict_testonly.py` 只是在本进程内把 `ConcatDataModule.setup()` 改成只构建 test
  fold，跳过推理不会用到的 train/dev 预处理；**没有修改仓库代码，也没有改变 test fold
  的构造和预测结果**，与直接运行 `multivers/predict.py` 的预测一致。
- batch size 不由命令行决定：`predict.py` 会重新读取 `../train_configs.json` 构造
  data module，实际使用其中的 `eval_batch_size=2`。`--batch_size` 对 data module 不生效。
- `--corpus_file` 在 `predict.py` 中实际未被使用；候选文本来自 `data_train` 的预处理结果。
- 实测：41,607 个 pair，单卡 RTX 3090，约 **53 分钟**（progress bar 3082 秒），
  峰值显存约 **19,760 MiB**（batch 2）。

### 1.6 随机种子

| 环节 | 种子/随机性 |
|---|---|
| 本次推理 | **没有设置随机种子，也不需要**。`model.eval()` + `model.freeze()`，无 dropout；test dataloader `shuffle=False`；无采样。 |
| Bootstrap 置信区间 | `seed=42`，2000 次重采样，重采样单位是 citation，n=605 |
| 作者训练脚本（仅背景） | `multivers/train.py` 中 `pl.seed_everything(76)`；本次实验**未重新训练**，该 seed 不影响本报告结果 |

推理本身不含随机操作；不同 GPU/库版本的浮点归约可能有极小数值差异，但不会改变标签。

### 1.7 评测脚本与评测口径

评测脚本：`eval_predictions.py`（仓库原本没有官方评测脚本，此脚本按论文 §2.6 的
P/R/F1、micro-F1、macro-F1 定义实现）。

命令：

```bash
cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode
python eval_predictions.py \
  --predictions predictions/test_predictions_official.jsonl \
  --claims data_train/target/scifact_10/claims_test.jsonl \
  --annotations ../Data/annotations.zip \
  --output predictions/eval_official.json
```

**关键口径**：

1. `label_probs` 的顺序是 `[P(NOT_ACCURATE), P(NEI/IRRELEVANT), P(ACCURATE)]`。
   模型输出里的 `NEI` 在评测中统一归一化为 `IRRELEVANT`。
2. **Citation-level 聚合规则**：对每条 citation，在全部候选文档中选
   `P(NOT_ACCURATE) + P(ACCURATE)` 最大的那个候选，然后取该候选的预测标签；
   如果该候选预测为 IRRELEVANT，则 citation 预测为 IRRELEVANT。
3. **Citation-level gold 来自 `Data/annotations.zip`**，按 citation context 文本匹配，
   而不是直接读 `claims_test.jsonl` 的 `evidence`。原因是 ETIQUETTE / IRRELEVANT
   citation 在 claims 文件里没有 evidence 条目，但它们在 annotations 里仍有
   citation-level 标签。
4. 606 条 test claims 中有 1 条无法一一匹配（id=106）：同一个 `<|multi_cit|>` 上下文
   对应了两条标注，fined label 分别是 ETIQUETTE 和 ACCURATE。评测脚本跳过该条，
   因此 citation-level n=605。这是确定性的文本匹配冲突，不是模型错误。
5. Pair level 直接把每个 claim–candidate abstract pair 的预测和 gold 比较。
   ETIQUETTE / IRRELEVANT citation 在 pair level 的 gold 全是 IRRELEVANT
   （因为没有 positive evidence 条目），这与 citation-level gold 不同，读数时要注意。

### 1.8 结果文件清单

| 文件 | 内容 | SHA256 |
|---|---|---|
| `predictions/experiment_metadata.json` | 实验来源、checkpoint、数据、环境、命令的机器可读记录 | `7fa76907aa8e10f8e8c3267e0257d9fb364e957a5445d6cb731a691cddc354de` |
| `predictions/test_predictions_official.jsonl` | 606 条 claim 的原始预测（4.4 MB） | `e1dc68fbb8867a1c1e0fb800d9c02a144c75f22337ca3046babe7db760318bba` |
| `predictions/eval_official.json` | citation/pair 两级指标 | `2ed183c2412cd08e2ebeb560ad31f18929c2852eb9f8b65b311375ce41bd6ff7` |
| `predictions/bootstrap_ci.json` | micro/macro 的 95% bootstrap CI | `5e562104b963c1ae2b4a46b9432c0067aac3d8196477aebc7ecfdbea17584060` |
| `predictions/error_analysis.json` | 605 条记录 + 231 条错误 + 混淆矩阵 | `4e3773e4edd97c953ec7fcfcfded058fd6a36dfe12444056a1093732bc5e12bb` |
| `predictions/repro_working_tree.diff` | 工作区 tracked diff | `af4667b63a8f01530b9561c35a82a5abffa5d16e92ea96686990385c205afe4d` |
| `eval_predictions.py` | 评测脚本 | `538d275ceed22efa0b18a2a794714eddd7624cf5edd6cec5a472bf3bc503b774` |
| `run_predict_testonly.py` | test-only 推理包装脚本 | `abe276d798aead78c9b41f5874633511be53963b5d41d8636a0d12e69f7895f3` |
| `checkpoints_official/bestModel.ckpt` | 作者 checkpoint | `b77f7afc6eeba2d137189b1c91d3c0baf51b8966af93cfd0326777207fc44bf4` |

### 1.9 Citation-level 主结果

**Per-class Precision / Recall / F1 / Support**

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ACCURATE | 0.7113 | 0.7850 | 0.7463 | 386 |
| NOT_ACCURATE | 0.3819 | 0.3254 | 0.3514 | 169 |
| IRRELEVANT | 0.4571 | 0.3200 | 0.3765 | 50 |
| **micro-F1 / macro-F1** | — | — | **0.6182 / 0.4914** | 605 |

> 单标签多分类下 micro-F1 = accuracy = 374/605 = 0.6182。

**Confusion matrix（行 = gold，列 = prediction）**

| gold \ pred | ACCURATE | NOT_ACCURATE | IRRELEVANT | 行合计 |
|---|---:|---:|---:|---:|
| ACCURATE | **303** | 73 | 10 | 386 |
| NOT_ACCURATE | 105 | **55** | 9 | 169 |
| IRRELEVANT | 18 | 16 | **16** | 50 |
| 列合计 | 426 | 144 | 35 | 605 |

**对应的 JSON（节选自 `predictions/eval_official.json`）**

```json
{
  "citation_level": {
    "num_examples": 605,
    "accuracy": 0.6181818181818182,
    "micro_f1": 0.6181818181818182,
    "macro_f1": 0.49140456887833905,
    "per_class": {
      "ACCURATE": {"precision": 0.7112676056338029, "recall": 0.7849740932642487, "f1": 0.7463054187192119, "support": 386},
      "NOT_ACCURATE": {"precision": 0.3819444444444444, "recall": 0.3254437869822485, "f1": 0.3514376996805112, "support": 169},
      "IRRELEVANT": {"precision": 0.45714285714285713, "recall": 0.32, "f1": 0.37647058823529417, "support": 50}
    },
    "confusion_matrix": {
      "ACCURATE": {"ACCURATE": 303, "NOT_ACCURATE": 73, "IRRELEVANT": 10},
      "NOT_ACCURATE": {"ACCURATE": 105, "NOT_ACCURATE": 55, "IRRELEVANT": 9},
      "IRRELEVANT": {"ACCURATE": 18, "NOT_ACCURATE": 16, "IRRELEVANT": 16}
    }
  }
}
```

### 1.10 与论文、基线的对比

| 指标 | 本实验（作者 ckpt） | 论文报告 | 差异 |
|---|---:|---:|---:|
| citation micro-F1 | **0.6182** | 0.59 | +0.0282 |
| citation macro-F1 | **0.4914** | 0.52 | −0.0286 |

**95% bootstrap CI（citation 重采样，2000 次，seed=42）**

| 指标 | 点估计 | 95% CI | 论文值是否在 CI 内 |
|---|---:|---|---|
| micro-F1 | 0.6182 | [0.5785, 0.6579] | **是**（0.59） |
| macro-F1 | 0.4914 | [0.4338, 0.5453] | **是**（0.52） |

**平凡基线（全猜 ACCURATE）**：micro-F1 = 386/605 = **0.6380**，macro-F1 = **0.2597**。
模型的 micro 略低于多数类基线（−0.020），但 macro 接近翻倍（0.4914 vs 0.2597）。
这与论文中“不能只看 micro，少数类才是关键”的模式一致。

**结论**：两个点估计与论文的差都在 ±0.03 以内，且论文值均落在 95% CI 内。按本实验的
评测口径，可以认为结果与论文报告**在统计误差范围内一致**。但这不是 Table 4 的严格复现，
原因见 §1.12。

### 1.11 Pair-level 参考结果

Pair level 共 41,607 个 claim–candidate abstract pair，其中 40,838 个（98.15%）gold 是
IRRELEVANT。因此 pair-level micro 主要反映“把负例判对”的能力，不能直接和 citation-level
的论文指标比较。

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ACCURATE | 0.0738 | 0.5048 | 0.1288 | 628 |
| NOT_ACCURATE | 0.0106 | 0.2766 | 0.0204 | 141 |
| IRRELEVANT | 0.9933 | 0.8181 | 0.8972 | 40,838 |
| **micro / macro** | — | — | **0.8116 / 0.3488** | 41,607 |

Pair-level 的 macro-F1 只有 0.349，说明模型在单个候选摘要上对 ACCURATE / NOT_ACCURATE
的判别很弱；citation-level 的 0.491 主要来自“在约 69 个候选里挑一个最像证据的摘要”这一步
聚合，而不是每个 pair 都判得很准。这也解释了为什么 pair-level 和 citation-level 的数字
看起来差别很大。

### 1.12 是否算“复现成功”？——严格 Table 4 的限制

**可以说的**：在作者发布的 checkpoint 和发布数据上，使用本报告重建的评测管线，
citation-level micro/macro 与论文报告值在 95% CI 内一致；错误结构也与论文的少数类
困难模式一致。

**不能说的**：这不是论文 Table 4 的严格复现。原因是输入构造不同：

1. 论文最优模型使用 **citance + BM25+MonoT5 检索的 top-20 sentences**，一个 citation
   一个样本（Table 4 设定）。
2. 发布数据给出的是 **每个 citation 约 69 个候选 abstract**，test 共 41,607 个
   claim–abstract pair；模型对每个候选单独打分，再由我们聚合到 citation。
3. checkpoint 内部 `num_training_instances=2454`、`global_step=1535`，反推训练时
   实例数在 2.4k 量级，对应每个 citation 约一个样本；这与发布数据的 197,418 个
   训练 pair 明显不同。
4. 论文的 BM25+MonoT5 top-20 数据没有随仓库发布。严格复现需要运行 `bm25t5.ipynb`
   （BM25 召回 + MonoT5 rerank，注意论文用 `TOP_n=20`，notebook 默认是 50），
   把每个 citation 的 top-20 句子拼成一个 evidence document，再按论文设定训练/推理。

因此，本报告应定位为：**“作者 checkpoint + 发布数据 + 重建评测管线”的一致性验证，
以及在此设定下的错误分析**，而不是“从零训练并严格复现 Table 4”。

---

## 2. 错误分析

### 2.1 总体错误结构

605 条 citation 中，231 条被错分，错误率 **38.2%**。

**按混淆矩阵单元拆分**

| gold → pred | 数量 | 占全部错误 | 该单元中有 gold evidence 的条数 | chosen doc ∈ gold evidence | 选中候选的中位 max prob | 中位 top1−top2 |
|---|---:|---:|---:|---:|---:|---:|
| NOT_ACCURATE → ACCURATE | **105** | **45.5%** | 61 | 16 | 0.932 | 0.865 |
| ACCURATE → NOT_ACCURATE | **73** | **31.6%** | 73 | 20 | 0.710 | 0.421 |
| IRRELEVANT → ACCURATE | 18 | 7.8% | 0 | 0 | 0.638 | 0.317 |
| IRRELEVANT → NOT_ACCURATE | 16 | 6.9% | 0 | 0 | 0.561 | 0.222 |
| ACCURATE → IRRELEVANT | 10 | 4.3% | 10 | 2 | 0.586 | 0.298 |
| NOT_ACCURATE → IRRELEVANT | 9 | 3.9% | 7 | 2 | 0.623 | 0.343 |

作为对照，正确预测的置信度整体更高（但 NOT_ACCURATE 内部相反：判错的反而更自信）：

| 正确单元 | 数量 | chosen doc ∈ gold evidence | 中位 max prob | 中位 top1−top2 |
|---|---:|---:|---:|---:|
| ACCURATE → ACCURATE | 303 | 99 | 0.974 | 0.949 |
| NOT_ACCURATE → NOT_ACCURATE | 55 | 8 | 0.680 | 0.401 |
| IRRELEVANT → IRRELEVANT | 16 | 0 | 0.714 | 0.534 |

**读法**：

- 最大的错误来源是 **NOT_ACCURATE → ACCURATE（105 条，45.5%）**，而且这些错误往往
  很“自信”（中位 max prob 0.932）。
- 第二大类是 **ACCURATE → NOT_ACCURATE（73 条，31.6%）**；其中 20 条选中的候选
  本身就是 gold evidence，说明不是“找不到证据”，而是“找到了却判成不支持”。
- `chosen doc ∈ gold evidence` 整体只有 147/605（24.3%）；在有 gold evidence 的
  486 条里也只有 147 条（30.2%）。这不等于 70% 的预测都找错了文档：gold evidence
  只标注了人工认定的那几段，同一结论可能出现在其他候选摘要里，模型选中别的摘要也可能
  是对的。但这个比例仍然说明**候选选择噪声很大**，尤其是在长摘要和多候选场景下。

### 2.2 按细粒度标签拆分

| 细粒度 gold | n | 其中无 gold evidence | pred ACCURATE | pred NOT_ACCURATE | pred IRRELEVANT | 该标签准确率 |
|---|---:|---:|---:|---:|---:|---:|
| ACCURATE | 370 | 0 | **290** | 70 | 10 | **78.4%** |
| ETIQUETTE | 72 | 69 | 46 | **24** | 2 | **33.3%** |
| IRRELEVANT | 50 | 50 | 18 | 16 | **16** | **32.0%** |
| OVERSIMPLIFY | 47 | 0 | 30 | **16** | 1 | **34.0%** |
| NOT_SUBSTANTIATE | 29 | 0 | 12 | **14** | 3 | **48.3%** |
| CONTRADICT | 12 | 0 | 11 | 0 | 1 | **0.0%** |
| MISQUOTE | 9 | 0 | 6 | **1** | 2 | **11.1%** |
| INDIRECT_NOT_REVIEW | 9 | 0 | 7 | 2 | 0 | **77.8%** |
| INDIRECT | 7 | 0 | 6 | 1 | 0 | **85.7%** |

**关键观察**：

1. **CONTRADICT 全军覆没**：12 条中 0 条判对，11 条判成 ACCURATE，1 条判成
   IRRELEVANT。模型几乎没有“方向/否定”能力。
2. **ETIQUETTE 和 IRRELEVANT 是重灾区**：两者合计 122 条，其中 119 条没有 gold
   evidence（ETIQUETTE 69/72，IRRELEVANT 50/50）。ETIQUETTE 有 46/72 被判成
   ACCURATE；IRRELEVANT 有 34/50 被判成非 IRRELEVANT。模型无法从“没有证据”的文本
   推断出标注约定。
3. **OVERSIMPLIFY / MISQUOTE 大部分被判成 ACCURATE**：OVERSIMPLIFY 30/47、
   MISQUOTE 6/9。这两类都是“文本有重叠，但 claim 省略了限定条件或改变了措辞”，
   模型对“省略/弱化”不敏感。
4. **NOT_SUBSTANTIATE 相对好一些**：14/29 判对，但仍有 12 条判成 ACCURATE、
   3 条判成 IRRELEVANT；模型在“相关但不充分”和“完全无关”之间摇摆。
5. **ACCURATE 是模型最擅长的类**：290/370 判对（78.4%），recall 0.785、precision
   0.711；但仍有 70 条被误判成 NOT_ACCURATE，说明模型对 claim 中的额外限定词偏严格。

**105 条 NOT_ACCURATE → ACCURATE 的精确分解**：

| 细粒度标签 | 数量 |
|---|---:|
| ETIQUETTE | 46 |
| OVERSIMPLIFY | 30 |
| NOT_SUBSTANTIATE | 12 |
| CONTRADICT | 11 |
| MISQUOTE | 6 |
| 合计 | **105** |

也就是说，这个最大的错误单元不是某一种标签造成的，而是**所有“细粒度负例”都被
模型倾向性地读成了支持**。

### 2.3 置信度与校准

这里的“置信度”是 citation-level 聚合选中的那个候选的 max `label_prob`，不是经过
校准的 citation-level 概率。按 max prob 分桶：

| 选中候选 max prob | n | citation accuracy |
|---|---:|---:|
| [0.5, 0.6) | 83 | 44.6% |
| [0.6, 0.7) | 68 | 52.9% |
| [0.7, 0.8) | 70 | 64.3% |
| [0.8, 0.9) | 66 | 57.6% |
| [0.9, 1.0) | 294 | 71.1% |

整体上高置信度更准，但并非单调（0.8–0.9 桶略低于 0.7–0.8 桶，样本也较小）。
更严重的是**按预测类别看，置信度并不可靠**：

| 预测类别 | max prob ≥ 0.9 的条数 | 其中正确 | Precision |
|---|---:|---:|---:|
| ACCURATE | 262 | 199 | **0.760** |
| NOT_ACCURATE | 23 | 5 | **0.217** |
| IRRELEVANT | 9 | 5 | **0.556** |

**高置信度的 NOT_ACCURATE 反而最不可信**（23 条里只有 5 条对）。更直接地说，
NOT_ACCURATE 判错的样本中位 max prob 是 0.932，反而高于判对样本的 0.680（§2.1）。
这意味着不能把 `P(NOT_ACCURATE)` 高当成“可以自动接受”的信号。相对地，高置信度
ACCURATE 的 precision 0.760，仍不足以完全自动接受，但可以作为人工复核的优先级排序。

### 2.4 典型错误样本

以下样本全部来自 `predictions/error_analysis.json` 的真实记录；`probs` 的顺序是
`[P(NOT_ACCURATE), P(IRRELEVANT/NEI), P(ACCURATE)]`。为控制篇幅，长摘要只引用
关键句，`…` 表示省略。

#### 案例 1：CONTRADICT → ACCURATE（id=176）

> **引用原文**：Prior data indicated that ACE2 expression is upregulated in certain group of asthma patients such as men, African Americans, diabetes, and those who are on prolonged use of corticosteroids (`<|cit|>`).
>
> **Gold evidence（doc 22004, sentence 2）**：Use of inhaled corticosteroids (ICS) was associated with lower expression of ACE2 and TMPRSS2, but treatment with triamcinolone acetonide did not decrease expression of either gene.
>
> **模型选中的候选（doc 22061）**：In summary, in a large cohort of well-characterized patients with asthma, we report higher sputum cell expression of ACE2 and TMPRSS2 in males, African Americans, and patients with diabetes mellitus and lower expression in patients taking inhaled corticosteroids. …
>
> **模型预测**：`[0.0009, 0.0000, 0.9991]` → **ACCURATE**

**为什么错**：claim 说“使用糖皮质激素的患者中 ACE2 上调”，而证据句和模型选中的摘要
都明确说“使用吸入性糖皮质激素与 **lower** expression 相关”。这是方向性矛盾，不是
“证据不足”。模型仍然给出 0.9991 的 ACCURATE，最可能的原因是 claim 和摘要共享大量
实体/短语（ACE2、TMPRSS2、asthma、males、African Americans、diabetes、
corticosteroids），模型把**词面重叠**当成了支持，没有检查“upregulated vs lower”的
方向。CONTRADICT 在 test 中只有 12 条，且 3 类折叠后没有专门的“矛盾”输出，模型很可能
没有学到足够的反向证据模式。

#### 案例 2：OVERSIMPLIFY → ACCURATE（id=13）

> **引用原文**：Examining the amount of genetic risk effect attributable to these genes (other than APOE), the most strongly associated single-nucleotide polymorphisms at each locus have population attributable fractions between 2.72 and 5.97%, with a cumulative population attributable fraction for non-APOE loci estimated to be as much as 35% (Naj et al., `<|cit|>`).
>
> **Gold evidence（doc 4031, sentence 2）**：… the most strongly associated SNPs at each locus other than APOE demonstrated population attributable fractions (PAFs) between 2.72–5.97% (Supplemental Table 9), with a cumulative PAF for non-APOE loci estimated to be as much as 35%; **however, these estimates may vary widely between studies22, and the actual effect sizes are likely to be much smaller than those estimated here because of the ‘winner’s curse’.**
>
> **模型选中的候选（doc 4031，恰好是 gold doc）**：… the most strongly associated SNPs at each locus other than APOE demonstrated population attributable fractions (PAFs) between 2.72–5.97% … with a cumulative PAF for non-APOE loci estimated to be as much as 35%; however, these estimates may vary widely between studies …
>
> **模型预测**：`[0.0060, 0.0001, 0.9939]` → **ACCURATE**

**为什么错**：claim 几乎逐字复制了数字，但删掉了“however … winner’s curse … 实际效应
可能小得多”的限定。OVERSIMPLIFY 的定义正是**省略限定/把有条件结论说成无条件结论**。
模型看到的是高度重叠的数字和术语，因此判为支持；它没有机制去检测“claim 中缺失了
证据中的 caveat”。注意模型选中的就是 gold doc，所以这不是检索错误，而是**标签边界
错误**：pair-level 的“文本支持”与 citation-level 的“是否过度简化”不是同一个判断。
OVERSIMPLIFY 47 条中有 30 条被判成 ACCURATE，说明这是系统性问题。

#### 案例 3：ETIQUETTE → ACCURATE（id=8）

> **引用原文**：The effect of the microbiome on atherosclerosis, glucose homeostasis and adiposity has been attributed, in part, to circulating metabolites such as trimethylamine-oxide and short-chain fatty acids (SCFAs) produced by gut bacteria`<|multi_cit|>`.
>
> **Gold**：ETIQUETTE（citation-level 折叠为 NOT_ACCURATE）。Gold evidence docs 10001 / 10003；claims 文件中该 citation **没有 positive evidence 条目**（72 条 ETIQUETTE 中有 69 条如此）。
>
> **模型选中的候选（doc 10008）**：Given the similarity in structure between L-carnitine and choline (Fig. 1a) we hypothesized that dietary L-carnitine in humans, like choline and phosphatidylcholine, might both produce TMA and TMAO in a gut microbiota-dependent fashion and be associated with atherosclerosis risk. …
>
> **模型预测**：`[0.0673, 0.0002, 0.9325]` → **ACCURATE**

**为什么错**：这是一条泛泛的背景陈述，引用的论文被用作“背景/礼节性引用”，而不是
用来支持这个具体断言；标注因此把它归为 ETIQUETTE（NOT_ACCURATE）。模型选中的摘要
确实在讲“L-carnitine/choline → TMAO → atherosclerosis”，主题高度相关，但它是**假设**
（`we hypothesized`）而不是对 claim 的证据。更根本的是：ETIQUETTE 是标注约定，
文本里没有“这句话只是礼节性引用”的信号；而且 pair-level 训练数据中这类 citation
根本没有 positive evidence，模型没有学过“输出 NOT_ACCURATE”的监督信号。
46/72 的 ETIQUETTE 被判成 ACCURATE，单这一项就贡献了 105 条最大错误单元中的 46 条。

#### 案例 4：NOT_SUBSTANTIATE → IRRELEVANT（id=42）

> **引用原文**：Increased incretin sensitivity improves β-cell function (`<|other_cit|>`), thereby lowering T2D risk (`<|cit|>`).
>
> **Gold evidence（doc 7039, sentences 1–2）**：We used the published MAGIC meta-analysis (up to 37,037 non-diabetic individuals) of HOMA indices of beta-cell function and insulin sensitivity5 … / The risk allele at ANK1 has features (nominally significant reduction in HOMA-B) indicating a primary effect on β-cell function, whereas those at GRB14 and AKNRD55 are characteristic of loci acting primarily through insulin resistance (increased HOMA-IR) …
>
> **模型选中的候选（doc 7011）**：aAlleles are aligned to the forward strand of NCBI Build 36.
>
> **模型预测**：`[0.0090, 0.9643, 0.0267]` → **IRRELEVANT**

**为什么错**：候选池里混入了大量表格脚注/方法学碎片（`aAlleles are aligned…` 只有
一句话），模型对这类文本给出很高的 IRRELEVANT 概率；而 gold doc 7039 讨论的是 HOMA
指标与 β-cell function，虽然相关，但并没有直接证明“incretin sensitivity 改善 β-cell
function”。模型在“相关但不充分（NOT_SUBSTANTIATE）”和“完全无关（IRRELEVANT）”之间
没有区分能力：它找不到直接支持，就退化成 IRRELEVANT。这个案例同时暴露了
**候选质量问题**（表格文本进入候选池）和**标签概念混淆**（NOT_SUBSTANTIATE vs
IRRELEVANT）。NOT_SUBSTANTIATE 29 条中，14 条判对，12 条判成 ACCURATE，3 条判成
IRRELEVANT。

#### 案例 5：ACCURATE → NOT_ACCURATE（id=7）

> **引用原文**：Prevotella-dominated healthy omnivore individuals were recently reported to have increased basal levels of serum TMAO (trimethylamine N-oxide), a product of inflammation linked to atherogenesis, compared to Bacteroides-dominated healthy individuals (`<|cit|>`).
>
> **Gold evidence（doc 10020, sentences 0–1）**：Several bacterial taxa remained significantly associated with plasma TMAO levels after false discovery rate (FDR) adjustment … / When subjects were classified into previously reported enterotypes18 … individuals with an enterotype characterized by enriched proportions of the genus Prevotella (n=4) demonstrated higher (p<0.05) plasma TMAO levels than subjects with an enterotype notable for enrichment of Bacteroides (n=49) genus (Fig. 2e).
>
> **模型选中的候选（doc 10020，恰好是 gold doc）**：When subjects were classified into previously reported enterotypes18 based upon fecal microbial composition, individuals with an enterotype characterized by enriched proportions of the genus Prevotella (n=4) demonstrated higher (p<0.05) plasma TMAO levels than subjects with an enterotype notable for enrichment of Bacteroides (n=49) genus (Fig. 2e).
>
> **模型预测**：`[0.8245, 0.0004, 0.1751]` → **NOT_ACCURATE**

**为什么错**：核心的 Prevotella vs Bacteroides / TMAO 结果在 gold 摘要里说得很清楚，
模型也选中了正确的摘要，但仍然判成 NOT_ACCURATE。最可能的解释是 claim 里多了
“healthy omnivore”和“a product of inflammation linked to atherogenesis”这两个限定：
摘要没有直接说“healthy omnivore”，也没有把 TMAO 称为“inflammation 的产物”。
模型可能把“claim 中有、摘要中没有”的修饰语当成不支持，而不是允许 claim 做合理的
背景补充。这是**“选对文档、判错标签”**的典型：73 条 ACCURATE → NOT_ACCURATE 中，
20 条选中的是 gold doc，70 条的细粒度 gold 就是 ACCURATE。模型缺少“部分支持/间接支持”
的概念，倾向于用严格的词面蕴含来判 NOT_ACCURATE。

#### 案例 6：ACCURATE → IRRELEVANT（id=39）

> **引用原文**：We identified 10 predominantly IR–related SNPs by 1) their significant association with HOMA-IR (P < 0.05) in the MAGIC data (`<|cit|>`).
>
> **Gold evidence（doc 7039, sentences 0–1）**：Second, we extended our previous analysis4 of the physiological consequences of T2D risk alleles to include the newly identified loci. / We used the published MAGIC meta-analysis (up to 37,037 non-diabetic individuals) of HOMA indices of beta-cell function and insulin sensitivity5 …
>
> **模型选中的候选（doc 7039，恰好是 gold doc）**：… We used the published MAGIC meta-analysis … of HOMA indices of beta-cell function and insulin sensitivity … The risk allele at ANK1 … whereas those at GRB14 and AKNRD55 are characteristic of loci acting primarily through insulin resistance (increased HOMA-IR) …
>
> **模型预测**：`[0.1817, 0.5472, 0.2711]` → **IRRELEVANT（NEI）**

**为什么错**：这是一条方法学/间接支持型 citation：claim 说“我们识别了 10 个 IR 相关
SNP”，gold 摘要说的是“用了 MAGIC 的 HOMA 指标、发现了与 insulin resistance 相关的
位点”，并没有在可见文本里逐字出现“10 个 SNP”。模型给出了最高的 NEI 概率（0.5472），
说明它没有把这种**方法学/间接支持**识别为 evidence。ACCURATE 类里包含
INDIRECT（7 条）和 INDIRECT_NOT_REVIEW（9 条），这类 citation 的 support 本身就是
间接的；模型倾向于要求 claim 和摘要之间有直接的断言对应，因此会把方法学引用判成
IRRELEVANT。10 条 ACCURATE → IRRELEVANT 数量不大，但方向一致。

#### 案例 7：IRRELEVANT → ACCURATE（id=45）

> **引用原文**：In a more recent large-scale meta-analysis which aimed to finely map the causal SNPs, only two of the 65 T2D susceptibility loci examined had a lead SNP resulting in a missense mutation (PPARG [rs1801282] and KCNJ11 [rs5215]) (`<|cit|>`).
>
> **Gold**：IRRELEVANT（该 citation 在 annotations 中没有 evidence 条目）。
>
> **模型选中的候选（doc 7061）**：We obtained summary statistics (RAFs, association P-values, allelic ORs and 95% CIs) for lead SNPs at the newly discovered loci in meta-analyses of T2D GWAS in: (i) 5,561 cases and 14,458 controls of South Asian descent … and (ii) 6,952 cases and 11,865 controls of East Asian descent …
>
> **模型预测**：`[0.0329, 0.0002, 0.9669]` → **ACCURATE**

**为什么错**：候选摘要在主题上和 claim 高度重叠（T2D、GWAS、lead SNP、meta-analysis），
但它讲的是“如何获取 summary statistics、样本量是多少”，并没有包含 claim 的具体事实
（65 个 loci 中只有 2 个 lead SNP 导致 missense mutation，PPARG/KCNJ11）。
IRRELEVANT 在这里的含义是“被引论文不包含这个具体结论”，而不是“论文和主题无关”。
模型把**主题相似**误当成**证据支持**，给出 0.9669 的 ACCURATE。18 条
IRRELEVANT → ACCURATE 很可能都是这个模式：claim 和被引论文属于同一研究领域，
但具体断言不被支持。

#### 案例 8：IRRELEVANT → NOT_ACCURATE（id=16）

> **引用原文**：All variants classified as “likely pathogenic” or with an “unknown pathogenicity”, and some “non-pathogenic” variants were genotyped in 961 sporadic AD cases, 1,346 unrelated elderly non-demented controls `<|multi_cit|>`
>
> **Gold**：IRRELEVANT（无 evidence）。
>
> **模型选中的候选（doc 4022）**：一段 GWAS 表格文本（`Gene:SNP| Cases| Controls| Total| ORM (95% CI)| …`）。
>
> **模型预测**：`[0.4993, 0.0043, 0.4963]` → **NOT_ACCURATE**（NOT_ACCURATE 只比
> ACCURATE 高 0.0030）

**为什么错**：这是一个**近似平局**。模型对表格文本没有把握，NOT_ACCURATE 和 ACCURATE
几乎各占一半；聚合规则选 `P(NOT)+P(ACC)` 最大的候选，然后取该候选的 argmax 标签，
于是 0.003 的差距决定了最终类别。这里有两个问题叠加：

1. 候选池里有表格/脚注文本，模型无法从中读出 claim 需要的证据；
2. citation-level 规则 `max P(NOT)+P(ACC)` 会**结构性压低 IRRELEVANT**：它先忽略
   IRRELEVANT 概率去挑“最像证据”的候选，再用该候选的 argmax 判类。当所有候选都
   不像证据时，规则仍然会给出一个非 IRRELEVANT 的标签，除非被选中的候选 argmax
   恰好是 IRRELEVANT。

605 条中有 45 条的 top1−top2 < 0.1；这些预测对聚合规则和浮点细节都很敏感。

#### 补充观察：选错候选也会很自信（id=4）

> **引用原文**：TMAO could be derived from increased consumption of meat (`<|cit|>`).
>
> **Gold evidence（doc 10001, sentence 1）**：Herein we demonstrate that intestinal microbiota metabolism of dietary L-carnitine, a trimethylamine abundant in red meat, also produces TMAO and accelerates atherosclerosis.
>
> **模型选中的候选（doc 10003）**：… Choline, a trimethylamine and part of the head group of phosphatidylcholine, is metabolized by gut microbiota to produce an intermediate compound known as trimethylamine (TMA) … TMA is rapidly further oxidized … to form trimethylamine-N-oxide (TMAO). …
>
> **模型预测**：`[0.9738, 0.0002, 0.0260]` → **NOT_ACCURATE**

模型选了一篇讲“choline → TMA → TMAO”的摘要，而不是讲“red meat 中的 L-carnitine →
TMAO”的 gold 摘要；它对前者给出 0.9738 的 NOT_ACCURATE。这个例子说明候选选择错误
会和过度自信叠加：即使 gold doc 10001 也在候选池里（模型给它 NOT_ACCURATE 0.757），
模型仍然更相信那篇“主题相近但没提 meat”的摘要。claim 里的“meat”是关键词，模型
似乎没有把它作为必须匹配的关系。

### 2.5 错误原因归纳

1. **语义方向/否定没有被建模。**
   CONTRADICT 12 条中 0 条判对，11 条判成 ACCURATE。案例 1 中 claim 说“upregulated”，
   证据说“lower expression”，模型仍然给出 0.9991 的 ACCURATE。高词面重叠压倒了
   方向判断；3 类折叠也没有提供专门的“矛盾”监督信号。

2. **“省略/弱化”类错误难以从文本对中检测。**
   OVERSIMPLIFY 30/47、MISQUOTE 6/9 被判成 ACCURATE。案例 2 中 claim 复制了数字但
   删掉了 caveat；模型看到的是高度重叠的文本，缺少“claim 少了什么”的表示。这类错误
   本质上是**claim 与证据之间的信息差**，不是简单的 entailment。

3. **标注约定没有对应的文本信号。**
   ETIQUETTE 69/72、IRRELEVANT 50/50 没有 gold evidence。ETIQUETTE 是“引用只用于
   背景/礼节”的约定，IRRELEVANT 是“被引论文不包含该具体结论”；两者都可能主题相关，
   但都不是“支持”。模型只能看到候选摘要，无法知道标注者的引用意图。ETIQUETTE
   46/72 判成 ACCURATE，是 105 条最大错误单元的最大来源。

4. **主题相似 ≠ 证据支持。**
   IRRELEVANT → ACCURATE 18 条。案例 7 中候选摘要和 claim 共享 T2D/GWAS/SNP/
   meta-analysis，但没有 claim 的具体事实，模型仍判 ACCURATE 0.9669。模型学到的
   更像“主题匹配”，而不是“具体断言是否被蕴含”。

5. **候选选择/检索噪声。**
   整体只有 147/605 的 chosen doc 是 gold evidence（有 gold evidence 的 486 条中
   30.2%）。案例 4 选中表格脚注，案例 8 选中表格文本，补充案例选中了主题相近但
   没提“meat”的摘要。候选池里有大量方法学碎片、表格和同主题摘要，pair-level 模型
   无法稳定挑出真正包含证据的那一篇。

6. **Citation-level 聚合规则会放大错误。**
   规则是“先选 `P(NOT)+P(ACC)` 最大的候选，再取 argmax”。这会：
   - 在近似平局时由 0.003 的差距决定标签（案例 8；45/605 条 top1−top2 < 0.1）；
   - 结构性压低 IRRELEVANT：即使所有候选都不像证据，规则也会尽量给一个非
     IRRELEVANT 的标签；
   - 把一个候选的高置信度当成整条 citation 的置信度，但候选选择和标签判断是两步，
     误差会累积。

7. **训练设定与推理设定不一致。**
   Checkpoint 的 `num_training_instances=2454`、`global_step=1535`（5 epochs，
   effective batch 8）说明作者训练时的实例数在 2.4k 量级，对应**每个 citation
   约一个实例**；本次推理是 41,607 个 claim–abstract pair，每个候选单独打分。
   模型没有在“多候选、长摘要、大量表格/方法学文本”的分布上训练过，候选选择噪声
   和 IRRELEVANT 抑制都可能与此有关。

8. **3 类折叠丢失了细粒度区分。**
   CONTRADICT、MISQUOTE、OVERSIMPLIFY、ETIQUETTE 全部折叠成 NOT_ACCURATE，
   模型没有输出“矛盾/误引/过度简化/礼节性引用”的能力。论文的任务本身就是 3 类，
   所以这不是实现错误，但它决定了模型不可能从输出层面区分这些错误类型；错误分析
   必须回到细粒度标签才能看清。

### 2.6 对“降低标注成本”的实际含义

如果目标是把这个模型当作人工标注前的预筛器，而不是自动标注器：

- **不要自动接受 NOT_ACCURATE。** 高置信度（≥0.9）的 NOT_ACCURATE precision 只有
  0.217（5/23）；整体 NOT_ACCURATE recall 也只有 0.325。把所有 NOT_ACCURATE 预测
  送人工复核，成本远低于漏掉错误引用。
- **高置信度 ACCURATE 可以作为优先队列，但不能直接终审。** ≥0.9 的 ACCURATE
  precision 0.760；仍有约 24% 是错的。适合“先看高置信度、再抽查低置信度”，不适合
  “高置信度直接通过”。
- **人工预算应优先给这些样本**：
  1. 所有预测为 NOT_ACCURATE 的 citation；
  2. 泛泛的背景陈述、`<|multi_cit|>`、ETIQUETTE 风格的 citation（模型最容易判成
     ACCURATE）；
  3. top1−top2 < 0.1 的低 margin 样本（45/605）；
  4. 选中候选是表格/方法学碎片、或所有候选的 `P(NOT)+P(ACC)` 都很低的 citation。
- **如果要做自动预标注，建议只自动接受“高置信度 ACCURATE + 候选集中 + 非 multi-cit”
  的小子集**，并在自己的数据上重新校准阈值；本报告的 0.760 precision 是
  scifact_10 test 上的结果，不能直接外推到其他领域。
- **矛盾/过度简化需要单独的检查器。** CONTRADICT 0/12、OVERSIMPLIFY 34% 准确率，
  说明单靠这个 3 类模型无法覆盖“引用完整性”的核心错误类型；可以叠加 NLI/LLM
  做 claim–evidence 的方向和 caveat 检查，或者针对这些类型增加细粒度监督。

### 2.7 本错误分析的局限

1. **不是严格 Table 4 设定**：checkpoint 训练时每个 citation 约一个实例，推理时是
   41,607 个 claim–abstract pair；论文的 BM25+MonoT5 top-20 数据未发布。
2. **评测脚本和聚合规则是重建的**，不是作者官方实现。换一种聚合规则（例如取最大
   P(ACCURATE)、对 `P(NOT)+P(ACC)` 设阈值、或对候选概率求和）会改变 citation-level
   数字，尤其是 IRRELEVANT 的 recall。
3. **1 条多引用上下文被跳过**（id=106，同一 `<|multi_cit|>` 上下文对应 ETIQUETTE 和
   ACCURATE 两条标注），因此 n=605 而不是 606。
4. **ETIQUETTE / IRRELEVANT 的 pair-level gold 全是 IRRELEVANT**（没有 positive
   evidence），但 citation-level gold 分别是 NOT_ACCURATE / IRRELEVANT；两个级别的
   标签含义不同，pair-level 指标不能直接解释成 citation-level 的 P/R/F1。
5. **`chosen_text` 在 `error_analysis.json` 中截断为 800 字符**，长摘要的引用只覆盖
   前 800 字符；案例讨论基于可见部分，未展示的后续句子可能影响模型判断。
6. **置信度未校准**：max `label_prob` 不是 citation-level 概率；按类别看，
   NOT_ACCURATE 的高置信度反而最不可靠。
7. **只跑了一个 test split，没有交叉验证**；bootstrap CI 只反映 605 条 citation 的
   抽样方差，不反映 checkpoint 训练方差或数据标注方差。
8. **没有重新训练**，因此无法判断这些错误是“这个 checkpoint 的问题”还是“3 类任务
   本身的不可分性”；要回答这个问题，需要在同一数据上训练多个 seed 并做 citation-level
   交叉验证。

---

## 3. 复现命令汇总

```bash
cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode
source /home/chenjw/anaconda3/etc/profile.d/conda.sh
conda activate citation-integrity

# 1) 推理（作者 checkpoint；test-only 预处理）
CUDA_VISIBLE_DEVICES=4 TOKENIZERS_PARALLELISM=false \
python run_predict_testonly.py \
  --checkpoint_path checkpoints_official/bestModel.ckpt \
  --input_file data_train/target/scifact_10/claims_test.jsonl \
  --corpus_file data_train/target/scifact_10/corpus.jsonl \
  --output_file predictions/test_predictions_official.jsonl \
  --device 0 --num_workers 4

# 2) 评测
python eval_predictions.py \
  --predictions predictions/test_predictions_official.jsonl \
  --claims data_train/target/scifact_10/claims_test.jsonl \
  --annotations ../Data/annotations.zip \
  --output predictions/eval_official.json
```

结果文件与 SHA256 见 §1.8；机器可读的来源记录见
`predictions/experiment_metadata.json`。
