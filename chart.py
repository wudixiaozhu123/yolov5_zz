import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取数据
data5 = pd.read_excel("data2.xlsx")

# 数据转换 (reshape/melt)
data5 = data5.melt(id_vars="model", var_name="Group", value_name="value")

# 设置 Treatment 和 Group 列的顺序
data5['model'] = pd.Categorical(data5['model'], categories=["yolov5s", "yolov5s+spp", "yolov5s+sppf", "yolov5s+MSFS-TFL"], ordered=True)
data5['Group'] = pd.Categorical(data5['Group'], categories=["mAP50", "Recall"], ordered=True)

# 设置绘图风格
sns.set(style="whitegrid", font_scale=1.3)

# 创建图形
plt.figure(figsize=(6, 6))

# 使用 Seaborn 绘制 Beeswarm plot
# sns.swarmplot() 用于绘制beeswarm plot（点图），显示每个数据点。
# x="Group" 设置 X 轴为“Group”列，y="value" 设置 Y 轴为“value”列。
# hue="Treatment" 按 Treatment 列对数据进行分组，自动为不同组别使用不同的颜色。
# dodge=True 使得不同处理（Treatment）的点分开绘制，避免重叠。
# size=6 设置数据点的大小。
# palette 定义了不同处理组的颜色。
ax = sns.swarmplot(x="Group", y="value", hue="model", data=data5,
                   dodge=True, size=6, palette={"yolov5s": "#4489C8", "yolov5s+spp": "#ED7E7A", "yolov5s+sppf": "#008F91", "yolov5s+MSFS-TFL": "#FFCD44"})

# 添加统计误差条和均值条，避免直接使用 `errorbar`
#sns.barplot() 用于绘制条形图，显示每个组的均值。
#estimator=np.mean 计算每个组别的均值作为条形图的高度。
#ci="sd" 计算标准差并显示为置信区间。
#dodge=True 保证不同 Treatment 的条形图分开绘制。
#color='white' 设置条形图的填充颜色为白色（背景透明）。
#edgecolor='black' 为条形图的边框设置黑色边缘。
#linewidth=1 设置边框的线宽为1。
#ax=ax 在先前创建的 ax 对象上绘制条形图。
sns.barplot(x="Group", y="value", hue="model", data=data5,
            estimator=np.mean, ci="sd", dodge=True, color='white', 
            edgecolor='black', linewidth=1, ax=ax)

# 添加分隔线
plt.axvline(x=1.5, linestyle='--', color='black', linewidth=0.8)

# 添加标注
annotations = [
    (0.8, 5100, "****"),
    (1.23, 5100, "****"),
    (1.8, 5100, "****"),
    (2.15, 5300, "****")
]

for x, y, label in annotations:
    ax.annotate(label, xy=(x, y), xytext=(x, y + 100),
                fontsize=14, color='black', ha='center', va='bottom')

# 设置图表的外观
ax.set_ylim(0, 150)#设置 Y 轴的显示范围
ax.set_yticks(np.arange(0, 151, 20))#设置 Y 轴的刻度间隔为 20，范围从 0 到 150。
ax.set_xlabel('')
ax.set_ylabel('value of evaluation indicators', fontsize=13)

# 移除 legend 的边框
ax.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)

# 设置标题
plt.title('')

# 保存图形
plt.tight_layout()
plt.savefig("BarDotErrorBar.pdf", format='pdf')

plt.show()
