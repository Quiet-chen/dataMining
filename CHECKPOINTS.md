# Checkpoints

> 这些文件体积很大（合计约 11.3GB），**不随代码包提交**。请按下面的链接单独下载，并放回指定路径。

推理需要 **两个** 文件：

1. `bestModel.ckpt` — 作者发布的 citation accuracy 模型；
2. `longformer_large_science.ckpt` — `model.py` 初始化 Longformer encoder 时加载。

`healthver.ckpt` 只在训练时作为起点，**推理不需要**。

---

## 1. bestModel.ckpt（推理主模型）

| 项目 | 内容 |
|---|---|
| 用途 | 作者发布的 citation accuracy 模型，推理主模型 |
| 大小 | 5,253,790,961 bytes（约 4.9 GiB） |
| SHA256 | `b77f7afc6eeba2d137189b1c91d3c0baf51b8966af93cfd0326777207fc44bf4` |
| 来源 | Google Drive 文件夹：https://drive.google.com/drive/folders/11b6Z8iv2FXObWmLaqfYzgUQsaL4QgTT2 |
| Google Drive file id | `1fnNiY_ECuOgDt561S6pmc8hzI4rnjn5Y` |
| 放置到 | `CitationAccuracyClassificationCode/checkpoints_official/bestModel.ckpt` |

下载示例（需要能访问 Google Drive）：

```bash
cd CitationAccuracyClassificationCode
mkdir -p checkpoints_official

pip install gdown==4.5.4
gdown --id 1fnNiY_ECuOgDt561S6pmc8hzI4rnjn5Y \
  -O checkpoints_official/bestModel.ckpt
```

校验：

```bash
sha256sum checkpoints_official/bestModel.ckpt
# 应输出：
# b77f7afc6eeba2d137189b1c91d3c0baf51b8966af93cfd0326777207fc44bf4
```

如果 Google Drive 无法直接访问，需要自行配置代理；不要把代理地址或凭据写进仓库。

---

## 2. longformer_large_science.ckpt（encoder 初始化）

| 项目 | 内容 |
|---|---|
| 用途 | 初始化 Longformer encoder；推理和训练都需要 |
| 大小 | 1,646,843,693 bytes（约 1.5 GiB） |
| 来源 | MultiVerS 官方 S3 |
| 下载 | https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/longformer_large_science.ckpt |
| 放置到 | `CitationAccuracyClassificationCode/checkpoints/longformer_large_science.ckpt` |

```bash
cd CitationAccuracyClassificationCode
mkdir -p checkpoints

wget -O checkpoints/longformer_large_science.ckpt \
  https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/longformer_large_science.ckpt
```

> `checkpoints/fast_download.sh` 是当时为了加速而写的脚本，里面固定了一个 S3 IP；换网络后可能失效，不建议作为唯一方式。优先使用上面的官方 URL。

---

## 3. healthver.ckpt（仅训练需要）

| 项目 | 内容 |
|---|---|
| 用途 | 训练起点；**推理不需要** |
| 大小 | 5,253,768,926 bytes（约 4.9 GiB） |
| 下载 | https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/healthver.ckpt |
| 放置到 | `CitationAccuracyClassificationCode/checkpoints/healthver.ckpt` |

```bash
cd CitationAccuracyClassificationCode
mkdir -p checkpoints

wget -O checkpoints/healthver.ckpt \
  https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/checkpoints/healthver.ckpt
```

也可以使用原仓库脚本：

```bash
cd CitationAccuracyClassificationCode
python get_checkpoint.py longformer_large_science
python get_checkpoint.py healthver
```

---

## 4. 目录检查

下载完成后，目录应类似：

```text
CitationAccuracyClassificationCode/
├── checkpoints/
│   ├── longformer_large_science.ckpt   1.5 GiB
│   └── healthver.ckpt                 4.9 GiB（仅训练）
└── checkpoints_official/
    └── bestModel.ckpt                  4.9 GiB
```

确认没有把 checkpoint 提交进 Git：

```bash
find . -type f -size +90M
```

如果输出里出现 `.ckpt`，说明提交包有问题；这些文件应通过链接分发，而不是放进仓库或课程 zip。
