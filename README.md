# yolov5s应用于CIFAR10分类任务

## 目录
- [安装](#安装)
- [使用](#使用)
- [贡献](#贡献)

## 安装

### 前提条件

列出所需的依赖项、环境或工具：

gitpython>=3.1.30</br>
matplotlib>=3.3</br>
numpy>=1.23.5</br>
opencv-python>=4.1.1</br>
Pillow>=9.4.0</br>
psutil  # system resources</br>
PyYAML>=5.3.1</br>
requests>=2.23.0</br>
scipy>=1.4.1</br>
thop>=0.1.1  # FLOPs computation</br>
torch>=1.8.0  # see https://pytorch.org/get-started/locally (recommended)</br>
torchvision>=0.9.0</br>
tqdm>=4.64.0</br>
ultralytics>=8.0.232</br>

Tip:下载项目后，使用pip install -r requirements.txt安装对应依赖即可

指路-->
优化器替换-yolov5-master下的train.py文件</br>
在此处选择合适的优化器，parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="Adam", help="optimizer")
在此处配置对应超参数，parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
在此处查看当前网络，parser.add_argument("--cfg", type=str, default="C:\\yolov5-master\\models\\yolov5s.yaml", help="model.yaml path")

关于核心代码部分的介绍与使用:
此部分等项目更新后会放出，因为项目目前再跑另外一个实验，代码部分还没有整合
