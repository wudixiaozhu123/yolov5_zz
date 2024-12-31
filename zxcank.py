import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches

# 读取数据
data1 = pd.read_excel("data.xlsx", sheet_name="b_WC_Smoking")
data1 = pd.melt(data1, id_vars=["Day", "Treatment"], value_name="value")

# 将数据类型转换为数字
data1["value"] = pd.to_numeric(data1["value"], errors='coerce')

# 计算所需的平均值
a = data1[(data1["Day"] == 35) & (data1["Treatment"] == "SMK")]["value"].mean()
b = data1[(data1["Day"] == 35) & (data1["Treatment"] == "NS+abx")]["value"].mean()
c = data1[(data1["Day"] == 35) & (data1["Treatment"] == "SMK+abx")]["value"].mean()

# 创建折线图
plt.figure(figsize=(10, 6))
sns.lineplot(x="Day", y="value", hue="Treatment", data=data1, marker="o", ci="sd")

# 添加背景色和竖线
plt.axvspan(21, 40, color='grey', alpha=0.3)
plt.axvline(x=21, linestyle="--", color="black")

# 设置标题和标签
plt.title("Weight Change (%)")
plt.xlabel("Day")
plt.ylabel("Weight change (%)")

# 设置图例和颜色
sns.set_palette("Set2")
plt.legend(title="", loc="upper left")

# 绘制标注
plt.annotate('***', xy=(37.5, 24), xytext=(37.5, 24),
             fontsize=14, color='black', rotation=90, ha="center")
plt.annotate('****', xy=(39.5, 28), xytext=(39.5, 28),
             fontsize=14, color='black', rotation=90, ha="center")

# 显示图像
plt.tight_layout()
plt.show()