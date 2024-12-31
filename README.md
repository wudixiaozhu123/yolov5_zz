# yolov5s应用于CIFAR10分类任务

## 目录
- [安装](#安装)
- [使用](#使用)
- [贡献](#贡献)

## 思路及安装

首先，思路是:yolov5s网络是有3个检测头的，分别对应小-中-大物体的检测然后再结合特征</br>
而我们知道CIFAR10数据集是32×32的，那么显然，如果要把他应用到yolov5中，删除中，大头，保留小头是一个好的选择。</br>

这边再提一下改进，下面简要陈列一下：</br>
1.对于检测部分，我改进了其yolo.py中的Detect类，删除掉了_make_grid方法，此方法用于生成 YOLO 等目标检测模型中的"网格"和"锚框"网格的工具函数。</br>
其作用是根据给定的网格尺寸和锚框的比例来构建用于目标检测的网格和锚框坐标。具体点儿就是，这个方法生成的网格用于将预测的坐标和对应的锚框进行匹配，从而定位物体在图像中的位置。</br>
那么由于是进行CIFAR10分类任务，那么这个方法存在的必要也不是很强了。</br>

2.其次，对于yolov5s-yaml网络结构，我们删除了中，大头的部分，仅保留小头，新的yaml文件为yolov5s-cf.yaml,在其中我们进行了对于的注释。</br>


对于加入了DualConv方法的网络，我们命名为DualConv.yaml，在其中，我们依然删除了中、大头。</br>
Concat 层：我们保持了 Concat 层的索引为 [-1, 6]，指向 backbone 中第6个层（即 P3/8），而去掉了其他无关的层。</br>
Detect 层：现在的 Detect 层只依赖于 P3/8（即 13 这个层），而去掉了 P4/16 和 P5/32。</br>

对于yolov5s-cf.yaml，我们的思路已经在其中进行了注释。</br>

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

Tips:请先安装GPU版本的torch,然后,使用pip install -r requirements.txt安装对应依赖即可

<H1>一些小提示</H1>
<font color=#FF7F50 size=7 face="黑体">配置部分在yolov5-master下的train.py文件中</br></font>
在此处选择合适的优化器，parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="Adam", help="optimizer")</br>
在此处配置对应超参数，parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")</br>
在此处查看当前网络，parser.add_argument("--cfg", type=str, default="C:\\yolov5-master\\models\\yolov5s.yaml", help="model.yaml path")</br>


