
---

````markdown
# 模型下载与安装指南（Linux）

> **硬件假设**：单卡 **NVIDIA TITAN V 12 GB**  
> **适用对象**：想要在本地或服务器上批量下载 MedSAM 系列模型（`MedSAM2`、`medsam-vit-base` 等）并用于 *Semi-SAM+* 训练 / 推理的研究与工程人员。

---

## 目录
1. [准备工作](#准备工作)  
2. [两种安装方式](#两种安装方式)  
   - 2.1 Python 脚本  
   - 2.2 Bash 脚本  
3. [跳过 LFS 加速安装](#跳过-lfs-加速安装)  
4. [后期拉取 LFS 大文件](#后期拉取-lfs-大文件)  
5. [目录结构与文件说明](#目录结构与文件说明)  
6. [常见问题 Troubleshooting](#常见问题-troubleshooting)  
7. [安装成功示例输出](#安装成功示例输出)  

---

## 准备工作

```bash
# 1. 安装 git-lfs
# ——Ubuntu / Debian——
sudo apt-get update && sudo apt-get install git-lfs
# ——CentOS / RHEL——
# sudo yum install git-lfs

# 2. 初始化 git-lfs
git lfs install
````

> **说明**：`git-lfs` 用于下载仓库中的大文件（权重、检查点等），若缺少会出现 “pointer file” 错误。

---

## 两种安装方式

> **默认下载路径**：`./yiming_models_hgf/`

### 2.1 使用 Python 脚本

```bash
# 先赋予可执行权限
chmod +x install_models.py

# A. 完整安装（推荐训练用）
python3 install_models.py

# B. 跳过 LFS 大文件（先搭环境，用时再下载）
python3 install_models.py --skip-lfs
```

### 2.2 使用 Bash 脚本

```bash
chmod +x install_models.sh

# A. 完整安装
./install_models.sh

# B. 跳过 LFS
./install_models.sh --skip-lfs
```

---

## 跳过 LFS 加速安装

`--skip-lfs` 仅拉取小文件（代码、config），大大加快首次 clone 速度；**训练前** 需手动拉取权重，方法见下一节。

---

## 后期拉取 LFS 大文件

```bash
cd yiming_models_hgf/MedSAM2
git lfs pull              # 拉取当前仓库所有大文件

cd ../medsam-vit-base
git lfs pull
```

若仓库为私有或 gated，需在 `git lfs pull` 时输入 Hugging Face **Access Token**，或先配置 SSH / HTTPS + Token（详见 Troubleshooting）。

---

## 目录结构与文件说明

```
yiming_models_hgf/
├── MedSAM2/              # 600 M 参数 - 记忆注意力版
│   ├── config.json
│   ├── pytorch_model.bin  # ~1 GB（LFS）
│   └── ...
└── medsam-vit-base/
    ├── config.json
    ├── pytorch_model.bin  # ~340 MB（LFS）
    └── ...
```

> `config.json` / `pytorch_model.bin` 即可直接被 `torch.load` / `transformers.AutoModel.from_pretrained` 调用。

---

## 常见问题 Troubleshooting

| 现象                                    | 解决办法                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------ |
| **`git: 'lfs' is not a git command`** | `sudo apt-get install git-lfs` 然后 `git lfs install`                      |
| **clone 卡在 \*.psd / \*.bin pointer**  | 忘记装 `git-lfs` 或使用了 `--skip-lfs`；进入仓库 `git lfs pull`                      |
| **Permission denied (publickey)**     | 改用 **HTTPS** URL + Access Token；或正确配置服务器 SSH key 并在 Hugging Face 账户里新增公钥 |
| **HTTP 403 / 401**                    | 仓库为私有 / gated，需要 Access Token；`git lfs pull` 时输入 token 作为密码              |
| **磁盘空间不足**                            | 删掉中间 `.git` 临时文件或只保留需要的 checkpoint；亦可挂载大容量盘                              |

---

## 安装成功示例输出<a id="安装成功示例输出"></a>

```text
MedSAM2 and MedSAM ViT Installation Script
==================================================
[INFO] Checking git-lfs installation...
[SUCCESS] git-lfs is installed

==================================================
Installing MedSAM2
==================================================
[INFO] Cloning MedSAM2 to yiming_models_hgf/MedSAM2...
[SUCCESS] MedSAM2 cloned with all files

==================================================
Installing MedSAM ViT Base
==================================================
[INFO] Cloning MedSAM ViT Base to yiming_models_hgf/medsam-vit-base...
[SUCCESS] MedSAM ViT Base cloned with all files

==================================================
INSTALLATION SUMMARY
==================================================
Base directory: /path/to/your/project/yiming_models_hgf
MedSAM2:        yiming_models_hgf/MedSAM2
MedSAM ViT Base: yiming_models_hgf/medsam-vit-base

[SUCCESS] Installation complete!
```
