# 复现步骤：Assessing Citation Integrity in Biomedical Publications (Bioinformatics 2024)

论文：Sarol et al., *Assessing citation integrity in biomedical publications: corpus
annotation and NLP models*, Bioinformatics 40(7), 2024, btae420.
仓库：<https://github.com/ScienceNLP-Lab/Citation-Integrity>

本机路径：

```
repo     = /home/chenjw/dataMining/Citation-Integrity
code     = $repo/CitationAccuracyClassificationCode
conda    = /home/chenjw/anaconda3
GPU      = 3 x RTX 3090 (sm_86, 24 GB each, currently shared)
```

---

## 0. 先看结论（重要）

1. **`bm25t5.ipynb` 不需要跑。** 发布的数据已经是 MultiVerS 格式：
   `Data/multivers-format/corpus.jsonl` 恰好是 `annotations/*/references_sentence/`
   里 100 篇 reference article 的 8515 个段落（已逐条核对），
   `claims-*.jsonl` 引用的是这些段落。BM25+T5 检索是论文里生成 top-k 句子的步骤，
   但发布数据已经给出段落级候选，直接训练即可。
2. **必须做标签映射。** 发布数据使用论文的细粒度标签
   （`ACCURATE / INDIRECT / CONTRADICT / ...`），而 `data_verisci.make_label()`
   和 `SciFactDataset` 只接受 MultiVerS 的 3 个类别，原样运行会直接
   `KeyError/ValueError`。映射表见第 3 节（`prepare_data.py` 已实现）。
3. **发布的数据是“引用-段落”级，而不是论文最优模型的“top-20 句子”级。**
   本仓库代码会对每篇 reference article 的每个段落各产生一个训练/预测样本
   （train 197,418 对，dev 24,797 对，test 41,607 对）。论文 Table 4 的最优模型
   输入是“citance + top-20 retrieved sentences”，一个 citation 一个样本。
   两者不完全等价，详见第 8 节。
4. **`predict.py` 依赖 `../train_configs.json`，但该文件在当前 HEAD 中被删除了。**
   可以从 git 历史恢复（见第 4 节）。
5. **`val_dataloader()` 实际用了 test fold（代码 bug）。** 训练时的
   `valid_*` 指标是在 test 上算的。若要干净的验证，需要一行修改（见第 6 节）。
6. 仓库没有提供推理结果的评测脚本。已补充 `eval_predictions.py`（见第 7 节）。

---

## 1. 创建环境

现有 `citation-integrity` conda 环境是空的 Python 3.9，可直接使用：

```bash
source /home/chenjw/anaconda3/etc/profile.d/conda.sh
conda activate citation-integrity

cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode
pip install -r requirements-repro.txt
```

`requirements-repro.txt` 里有两处针对旧依赖的必要修正：

- `PyYAML==6.0`：`pytorch-lightning==1.2.1` 的元数据明确写了
  `PyYAML (!=5.4.*,>=5.1)`，原始 `5.4.1` 的 pin 无法安装。
- `Werkzeug==1.0.1`：`tensorboard==2.5.0` 是 2021 年的版本，固定到同期的
  Werkzeug，避免新版本 dev-server API 变化带来的兼容问题。

为什么不用原始 `requirements.txt` 的 `torch==1.7.1`：

- RTX 3090 是 Ampere（`sm_86`）。
- PyPI 上 `torch==1.7.1` 默认是 CUDA 10.2 wheel，不包含 `sm_86` kernel，
  运行时会报 `no kernel image is available for execution on the device`。
- `torch==1.8.1+cu111` 是包含 `sm_86` 且仍兼容 `pytorch-lightning==1.2.1` 的版本。

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 期望: 1.8.1+cu111 True (8, 6)
```

### 1.1 HuggingFace 访问（本机需要）

本机直连 `huggingface.co` 不通，但 `hf-mirror.com` 可用。`transformers==4.2.2`
需要三处兼容补丁（URL 前缀、307 相对跳转、429 时的缓存回退），仓库里已提供脚本：

```bash
python patch_mirror_compat.py
```

另外，`transformers==4.2.2` 的 `LongformerTokenizerFast` 把 tokenizer 文件地址
硬编码成了 `https://huggingface.co/roberta-large/...`，不会走 mirror 前缀。
本机直连 HF 不通，因此需要把这三个文件按这些 URL 的缓存键放进 HF 缓存：

```bash
python - <<'PY'
import hashlib, os, requests
from transformers.file_utils import TRANSFORMERS_CACHE
base = "https://hf-mirror.com/allenai/longformer-large-4096/resolve/main/"
for name in ["vocab.json", "merges.txt", "tokenizer.json"]:
    url = "https://huggingface.co/roberta-large/resolve/main/" + name
    dst = os.path.join(TRANSFORMERS_CACHE, hashlib.sha256(url.encode()).hexdigest())
    r = requests.get(base + name)
    r.raise_for_status()
    open(dst, "wb").write(r.content)
    print("cached", name, dst)
PY
```

验证：

```bash
python -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('allenai/longformer-large-4096'); print('ok')"
```

`hf-mirror` 偶尔返回 429；上面的补丁会优先使用缓存。重新安装 transformers
后需要重新运行 `patch_mirror_compat.py`，并重放 tokenizer 缓存。

### 1.2 删除两个未使用的 import

`multivers/model.py` 顶部 import 了 `bertviz` 和 `focal_loss`，但代码里从未使用。
PyPI 的 `focal-loss` 是 TensorFlow 包，不提供 `focal_loss.focal_loss.FocalLoss`，
直接装还会踩坑。当前仓库里已经注释，下面命令可跳过：

```bash
sed -i 's/^from bertviz import head_view/# from bertviz import head_view/' multivers/model.py
sed -i 's/^from focal_loss.focal_loss import FocalLoss/# from focal_loss.focal_loss import FocalLoss/' multivers/model.py
grep -n "bertviz\|focal_loss" multivers/model.py   # 确认只剩注释
```

### 1.3 去掉刷屏的 debug 输出

`data_train.py` 会对每个样本打印 rationale 和 claim（近 20 万行）。当前仓库里
这两行 `print` 已经删除，下面命令可跳过：

```bash
sed -i '/print(rationales, claim)/d' multivers/data_train.py
```

---

## 2. 下载起始 checkpoint

README 里 `python get_checkpoint.py` 少了必需的位置参数，会直接报错。实际要指定名字：

```bash
cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode

python get_checkpoint.py healthver                  # 5.25 GB -> checkpoints/healthver.ckpt
python get_checkpoint.py longformer_large_science   # 1.65 GB -> checkpoints/longformer_large_science.ckpt
```

不要用 `all`，它会下载 7 个模型（十几 GB），这里只需要上面两个。
两个 S3 链接已验证可访问。

`healthver.ckpt` 是 MultiVerS 的预训练起点；`longformer_large_science.ckpt`
是 `model._get_encoder()` 硬编码加载的 Longformer 科学领域权重。

原版代码还会在首次运行时下载 `allenai/longformer-large-4096`（约 1.4 GB）：
tokenizer 读取 config/vocab/merges/tokenizer 文件，`LongformerModel.from_pretrained`
读取 `pytorch_model.bin`。按 1.1 配好 `hf-mirror` 后无需手工干预，文件会缓存到
`~/.cache/huggingface/transformers/`，之后离线可用。

---

## 3. 准备训练数据

已提供脚本：

```bash
python prepare_data.py
```

它会生成：

```
data_train/target/scifact_10/
├── claims_train.jsonl   # 2141 citations, 197418 (citation, paragraph) pairs
├── claims_dev.jsonl     #  316 citations,  24797 pairs
├── claims_test.jsonl    #  606 citations,  41607 pairs
├── corpus.jsonl         # 8515 paragraphs
└── label_report.json
```

（`data_train/` 已经生成并校验过：所有 `cited_doc_ids` 都能在 corpus 找到，
所有 rationale 句子下标都合法。）

### 3.1 标签映射（论文 Section 2.5.3）

| 细粒度标签 | 3 类标签 | 代码内类别 |
|---|---|---|
| `ACCURATE` | `ACCURATE` | 2 |
| `INDIRECT` | `ACCURATE` | 2 |
| `INDIRECT_NOT_REVIEW` | `ACCURATE` | 2 |
| `CONTRADICT` | `NOT_ACCURATE` | 0 |
| `NOT_SUBSTANTIATE` | `NOT_ACCURATE` | 0 |
| `OVERSIMPLIFY` | `NOT_ACCURATE` | 0 |
| `MISQUOTE` | `NOT_ACCURATE` | 0 |
| `ETIQUETTE` | `NOT_ACCURATE` | 0 |
| `IRRELEVANT` | `IRRELEVANT` | 1 |
| （未被引用的段落） | `IRRELEVANT` | 1 |

说明：

- 论文把 `INDIRECT` 并入 `ACCURATE`；发布数据里 `INDIRECT_NOT_REVIEW` 与
  `INDIRECT` 合计 82 条，正好等于论文 Table 1 的 `INDIRECT=82`，所以它也是
  `INDIRECT` 的子类，映射到 `ACCURATE`。
- 代码内部把类别 1 叫 `NEI`/`NOT ENOUGH INFO`，论文叫 `IRRELEVANT`，是同一类。
- 细粒度 `IRRELEVANT` 的 citation 通常没有 evidence segment；脚本会把这类
  evidence 条目删掉，于是它的所有段落都成为类别 1。数据集里只有 2 条（train 1、
  test 1）带 evidence 的 `IRRELEVANT`，影响可忽略。
- 脚本保留原始文件不动，只写 `data_train/` 下的副本。

---

## 4. 恢复推理配置

`predict.py` 第 52 行硬编码读取 `../train_configs.json`，但训练代码里写这个文件的
语句被注释掉了，当前 HEAD 也删除了该文件。从 git 历史恢复作者当时的配置：

```bash
cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode
git show 0dbcf906e^:CitationAccuracyClassificationCode/train_configs.json > ../train_configs.json
python -c "import json; d=json.load(open('../train_configs.json')); \
print(d['datasets'], d['encoder_name'], d['max_label_weight'])"
# scifact_10 longformer-large-science 5.0
```

---

## 5. 冒烟测试（先确认整条链路能跑）

```bash
cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode
nvidia-smi                     # 先看哪张卡空闲
export CUDA_VISIBLE_DEVICES=1  # 例如用 1 号卡

CUDA_LAUNCH_BLOCKING=1 TOKENIZERS_PARALLELISM=false \
python train_target.py --dataset scifact_10 --gpus=1 --debug > debug.log 2>&1
```

`--debug` 让每个 fold 只读前 10 个 citation，但 MultiVerS 是**句级**样本
（每个 citation 的每句 evidence 都是一个训练实例），所以实际约 1135 个 train
batch / 445 个 val batch。不开启 gradient checkpointing 时一个 epoch 约 8 分钟；
显存紧张时加 `--gradient_checkpointing`（原版支持的参数），约 19 分钟一个 epoch。
看到第一次 `valid_*` 指标输出就说明数据、tokenizer、模型、checkpoint 都通了，
可以停止。

本机实测（RTX 3090，`CUDA_VISIBLE_DEVICES=4`，`--gradient_checkpointing`，
epoch 0 结束）：

```
valid_label_accuracy              = 0.8551
valid_sentence_label_f1           = 0.0702
保存 checkpoints_user/scifact_10_1/checkpoint/epoch=0-step=86.ckpt
```

`--debug` 只训练 10 个 citation，指标本身没有参考价值；这一步只验证链路。
确认后 Ctrl-C（或 `kill` 训练进程）释放显存即可。

注意：train 的前 10 个 citation 恰好没有 `NOT_ACCURATE` 样本，原版
`_add_instance_weights` 用 `.loc[[...]]` 会直接 `KeyError`。本仓库已改成
`reindex([...], fill_value=0).clip(lower=1)`，因此 `--debug` 可以正常跑；
完整训练时的权重与原来一致。

常见问题：

- 报 `python: command not found` 或用了系统 Python：`train_target.py` 用
  `subprocess` 调 `"python"`，必须确保 conda 环境已激活（`which python` 指向
  `anaconda3/envs/citation-integrity/bin/python`）。
- 报 `no kernel image`：torch 装成了 CUDA 10.2 版本，重装 `torch==1.8.1+cu111`。
- 报 HF 下载失败：重做 1.1 的镜像补丁，或重试。
- OOM：加 `--gradient_checkpointing`（`train_target.py` 已支持）。

---

## 6. 正式训练

```bash
cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode
export CUDA_VISIBLE_DEVICES=1
CUDA_LAUNCH_BLOCKING=1 TOKENIZERS_PARALLELISM=false \
python train_target.py --dataset scifact_10 --gpus=1 > output.txt 2>&1
```

实际传给 `multivers/train.py` 的关键参数（见 `train_target.py`）：

```
--starting_checkpoint checkpoints/healthver.ckpt
--encoder_name longformer-large-science
--lr 1e-5
--precision 16
--max_epochs 5
--accumulate_grad_batches 8   # effective batch size 8
--train_batch_size 1
--eval_batch_size 2
--no_reweight_labels          # 但 datamodule 内部又强制改成 False
```

输出：

```
checkpoints_user/scifact_10/checkpoint/   # top-20 + last checkpoints
checkpoints_user/scifact_10/              # TensorBoard/CSV logs
```

时间/显存预估（按本机 3090 实测吞吐）：

- train 197,418 对 / epoch，test 41,607 对 / epoch。
- 不开启 gradient checkpointing：约 2.2–2.7 步/秒 → **20–25 小时/epoch**，
  5 epoch 约 **4–5 天**。
- 开启 gradient checkpointing：约 1.0 步/秒 → **约 55 小时/epoch**，
  5 epoch 约 **11–12 天**。
- Longformer-large + 4096 token + batch 1 + fp16，通常占 12–20 GB；24 GB 卡够用。
  本机 GPU 多人共用，训练前先 `nvidia-smi` 选空闲卡。
- 建议放 `tmux`/`nohup`，并预留足够磁盘（每个 checkpoint 约 4.9 GB，
  top-20 + last 最多约 100 GB）。

### 6.1 两个已知代码问题

**a) 验证集用了 test。** `multivers/data_train.py` 的

```python
def val_dataloader(self):
    return self.get_dataloader("test", self.eval_batch_size)
```

而 `setup()` 其实也构建了 `"val"`（dev）。如果按正常流程用 dev 选模型：

```bash
sed -i 's/return self.get_dataloader("test", self.eval_batch_size)/return self.get_dataloader("val", self.eval_batch_size)/' \
  multivers/data_train.py
```

保持原样 = 完全按发布代码跑（checkpoint 选择会看到 test）；改成 `"val"` =
更规范的复现（用 10 篇 dev 文章选模型）。论文没有明确说明这一点，两种都建议记录。

**b) `--no_reweight_labels` 实际无效。** `ConcatDataModule.__init__` 里无条件执行
`hparams.no_reweight_labels = False`，随后 `reweight_labels = not ... = True`。
所以无论命令行是否传这个 flag，都会按标签频率加权。按发布代码原样保留即可。

---

## 7. 推理与评测

### 7.1 用作者发布的最优模型（推荐先走这条）

Google Drive 文件夹：
<https://drive.google.com/drive/u/0/folders/11b6Z8iv2FXObWmLaqfYzgUQsaL4QgTT2>

```bash
pip install gdown==4.5.4
gdown --folder "https://drive.google.com/drive/folders/11b6Z8iv2FXObWmLaqfYzgUQsaL4QgTT2" \
      -O checkpoints_official
ls checkpoints_official
```

若 Google Drive 不可达，就在能联网的机器上下载后拷过来。

推理（必须在 `CitationAccuracyClassificationCode/` 下运行，因为要读
`../train_configs.json`）：

```bash
mkdir -p predictions
python multivers/predict.py \
  --checkpoint_path checkpoints_official/<best>.ckpt \
  --input_file  data_train/target/scifact_10/claims_test.jsonl \
  --corpus_file data_train/target/scifact_10/corpus.jsonl \
  --output_file predictions/test_predictions.jsonl \
  --device 0 \
  --num_workers 4
```

注意 `predict.py` 的实际行为：

- **`--corpus_file` 没有被使用**；`--input_file` 只用来取 claim ID 列表用于输出对齐。
- 真正加载数据的是 `../train_configs.json` 指定的 `datasets=scifact_10`，
  即固定读取 `data_train/target/scifact_10/claims_test.jsonl` + `corpus.jsonl`
  （test fold）。所以 `--input_file` 要用同一个文件，否则 ID 对不上。
- `--device` 是相对 `CUDA_VISIBLE_DEVICES` 的编号，单卡设为 0 即可。
- 当前 `predict.py` 的输出只保留 `label` 和 `label_probs`，不输出 rationale
  （作者在提交历史里删掉了 `sentences` 字段）。

用自己训练的 checkpoint 时，把 `--checkpoint_path` 换成
`checkpoints_user/scifact_10/checkpoint/<best>.ckpt` 即可。

### 7.2 评测

仓库没有评测脚本，已补 `eval_predictions.py`：

```bash
python eval_predictions.py \
  --predictions predictions/test_predictions.jsonl \
  --claims data_train/target/scifact_10/claims_test.jsonl \
  --annotations ../Data/annotations.zip \
  --output predictions/eval_results.json
```

它会输出：

- **pair level**：对 test 的 41,607 个 (citation, paragraph) 对直接算 P/R/F1。
- **citation level**：每个 citation 选一个“最像 evidence”的段落
  （`P(ACCURATE)+P(NOT_ACCURATE)` 最大），用它预测 citation 标签；
  gold 用 `annotations.zip` 里的 citation 级细粒度标签（这样 `ETIQUETTE` /
  `IRRELEVANT` 这类没有 evidence 的 citation 也有正确 gold）。
- 混淆矩阵、per-class P/R/F1、micro-F1、macro-F1、细粒度标签的预测分布。

论文报告的是 **citation level**：最优模型 micro-F1 0.59、macro-F1 0.52
（Table 4）。pair level 因 98% 以上是 IRRELEVANT 负例，数字会高很多，不能直接和
论文比。

citation-level 聚合规则是我按任务语义补的（论文没有随仓库发布评测代码），
如果你有作者的原始评测脚本，以作者为准。

### 7.3 作者 `bestModel.ckpt` 实测结果（2026-10-28）

从 Google Drive 下载 `bestModel.ckpt`（4.9 GB），在当前段落级 test 上跑全量推理：

- 41,607 个 (citation, paragraph) pair，605 条可评 citation；单卡 RTX 3090，
  约 52 分钟，显存约 20 GB（batch 2）。
- citation level：micro-F1 **0.6182**，macro-F1 **0.4914**。
- 95% bootstrap CI（重采样 605 条 citation）：micro [0.579, 0.658]，
  macro [0.434, 0.545]。
- 论文 Table 4：micro 0.59、macro 0.52。两个值都落在 CI 内，差值约 0.03。
- per-class：ACCURATE F1 0.746（n=386）；NOT_ACCURATE 0.351（n=169）；
  IRRELEVANT 0.376（n=50）。
- pair level（仅参考，98% 以上是 IRRELEVANT 负例）：micro 0.812，macro 0.349。
- 平凡基线（全猜 ACCURATE）：micro 0.638、macro 0.260。模型 micro 略低于
  多数类基线，但 macro 接近翻倍，和论文的模式一致。

**严格对齐 Table 4 的限制**：checkpoint 的 hparams 里
`num_training_instances = 2454`、`global_step = 1535`（epoch 5），说明作者模型
是按“一个 citation 一个样本”的 top-x 设定训练的；当前 test 是 41,607 个
paragraph pair。因此上面的结果应看作“发布 checkpoint + 发布数据/评测管线”的
一致性验证，而不是 Table 4 的严格复现。严格复现仍需 `bm25t5.ipynb` 的
BM25+MonoT5 top-20 数据。

---

## 8. 发布仓库与论文的差异（复现时必须知道）

1. **top-20 sentences vs paragraphs。**
   论文最优模型：citance + BM25+MonoT5 检索的 top-20 句子，作为一个 evidence
   input，一个 citation 一个样本（Table 4）。
   发布数据：每篇 reference article 的所有段落都作为候选，train 有 197,418 个
   (citation, paragraph) 对。两者不是同一个任务设定，指标不能直接对齐。
   若要严格复现论文 Table 4，需要：
   - 用 `bm25t5.ipynb` 对每篇 reference article 建 Lucene index、BM25 召回、
     MonoT5 rerank（依赖 `pyserini`、`pygaggle`、Java、`castorini/monot5-base-med-msmarco`），
     注意 notebook 默认 `TOP_n=50`，论文用的是 **20**；
   - 把每个 citation 的 top-20 句子拼成一个 evidence document，构造
     `claims_*.jsonl` + `corpus.jsonl`（每个 citation 只引用这一个 doc）；
   - 再按本指南训练/推理。
2. **`ETIQUETTE` / `IRRELEVANT` citation 在段落级数据里没有 evidence。**
   3063 个 citation 中约 622 个没有 evidence 条目（其中大部分是 `ETIQUETTE`，
   论文把它们映射为 NOT_ACCURATE）。段落级模型只能把这些 citation 的所有段落
   当负例，学不到“错误引用”的信号。这是发布数据本身的限制，也会明显拉低
   NOT_ACCURATE 的 recall（见 `eval_predictions.py` 的 fine_grained_breakdown）。
3. **验证集用了 test**（第 6.1 节）。
4. **标签重加权 flag 被覆盖**（第 6.1 节）。
5. **发布数据是细粒度标签，代码只支持 3 类**，必须运行 `prepare_data.py`。
6. **没有官方评测脚本**；`metrics.py` 是训练时用的 PyTorch Lightning Metric，
   不直接吃 `predict.py` 输出的 JSONL。
7. 论文补充材料写 lr=5e-5，但发布的 `train_target.py` 用 `--lr 1e-5`，
   `train_configs.json` 也是 1e-5。按发布脚本跑。

---

## 9. 最短命令清单

```bash
source /home/chenjw/anaconda3/etc/profile.d/conda.sh
conda activate citation-integrity
cd /home/chenjw/dataMining/Citation-Integrity/CitationAccuracyClassificationCode

# 1) 环境（只需一次）
pip install -r requirements-repro.txt
# 1.1 的 HF 镜像补丁、1.2 的 import 注释、1.3 的去 print（只需一次）

# 2) checkpoint（只需一次）
python get_checkpoint.py healthver
python get_checkpoint.py longformer_large_science

# 3) 数据
python prepare_data.py

# 4) 配置
git show 0dbcf906e^:CitationAccuracyClassificationCode/train_configs.json > ../train_configs.json

# 5) 冒烟
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 TOKENIZERS_PARALLELISM=false \
python train_target.py --dataset scifact_10 --gpus=1 --debug > debug.log 2>&1

# 6) 正式训练
CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
python train_target.py --dataset scifact_10 --gpus=1 > output.txt 2>&1

# 7) 推理
python multivers/predict.py \
  --checkpoint_path checkpoints_user/scifact_10/checkpoint/<best>.ckpt \
  --input_file data_train/target/scifact_10/claims_test.jsonl \
  --corpus_file data_train/target/scifact_10/corpus.jsonl \
  --output_file predictions/test_predictions.jsonl \
  --device 0 --num_workers 4

# 8) 评测
python eval_predictions.py \
  --predictions predictions/test_predictions.jsonl \
  --output predictions/eval_results.json
```
