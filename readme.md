

# 项目名称：HIMPI


## 目录

- [项目名称](#项目名称)
  - [目录](#目录)
  - [环境要求](#环境要求)
  - [数据集](#数据集)
    - [数据集位置](#数据集位置)
    - [下载文件](#下载文件)
  - [训练及测试](#训练及测试)
    - [训练命令](#训练命令)
    - [测试命令](#测试命令)

## 环境要求

运行代码所需的软件和库：

- Python 3.7+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- 其他依赖（可通过 `pip install -r requirements.txt` 安装）

## 数据集

### 数据集位置

- 训练集 `./train/`
- 测试集 `./test/`

### 下载文件

训练集中的图像列表可在 `trainfile_list.txt` 文件中找到。

| 文件名 | 描述 | 下载链接 |
|--------|------|----------|
| trainfile_list.txt | 训练集 | [下载链接](https://lightfields.stanford.edu/LF2016.html) |
| test | 测试集 | [下载链接](https://bjtueducn-my.sharepoint.com/:f:/g/personal/23120336_bjtu_edu_cn/IgCYg-GcccJATrYWNXt-nhgEASSfOC5PBwAQjFg8T24jAaE?e=w9hnge) |
| test_challenging | 更挑战性测试集 | [下载链接](https://bjtueducn-my.sharepoint.com/:f:/g/personal/23120336_bjtu_edu_cn/IgDM-JgpFi4WQ4n16j1SglWCAU-T9aieFjbhU0geuOb5ptg?e=9nGgoN) |

下载后请将文件放置在 `./NetworkSave/%taskname%/` 目录下。

需要将[vgg19-dcbb9e9d.pth](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)直接放置在当前目录下.

## 训练及测试

### 训练命令

```bash
cd base
python train.py
```

### 测试命令

```bash
cd base
python test.py
```

---


# Project Name: HIMPI


## Table of Contents

- [Project Name: HIMPI](#project-name-himpi)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Dataset](#dataset)
    - [Dataset Location](#dataset-location)
    - [Download Files](#download-files)
  - [Training and Testing](#training-and-testing)
    - [Training Command](#training-command)
    - [Testing Command](#testing-command)

## Requirements

Software and libraries required to run the code:

- Python 3.7+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- Other dependencies can be installed via `pip install -r requirements.txt`

## Dataset

### Dataset Location

- Training set: `./train/`
- Test set: `./test/`

### Download Files

The list of images in the training set can be found in the file `trainfile_list.txt`.

| File Name | Description | Download Link |
|-----------|-------------|----------------|
| trainfile_list.txt | Training set | [Download Link](https://lightfields.stanford.edu/LF2016.html) |
| test | Test set | [Download Link](https://bjtueducn-my.sharepoint.com/:f:/g/personal/23120336_bjtu_edu_cn/IgCYg-GcccJATrYWNXt-nhgEASSfOC5PBwAQjFg8T24jAaE?e=w9hnge) |
| test_challenging | Challenging Test set | [Download Link](https://bjtueducn-my.sharepoint.com/:f:/g/personal/23120336_bjtu_edu_cn/IgDM-JgpFi4WQ4n16j1SglWCAU-T9aieFjbhU0geuOb5ptg?e=9nGgoN) |


After downloading, please place the files under the `./NetworkSave/%taskname%/` directory.
You need to place [vgg19-dcbb9e9d.pth](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) directly in the current directory.

## Training and Testing

### Training Command

```bash
cd base
python train.py
```

### Testing Command

```bash
cd base
python test.py
```
