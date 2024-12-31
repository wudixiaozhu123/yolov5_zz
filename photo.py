import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, text, x, y, width, height, fontsize=10):
    """绘制一个矩形框，并在其中添加文本"""
    box = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey")
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)

def draw_arrow(ax, x1, y1, x2, y2):
    """绘制箭头"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=8))

def main():
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # 添加模块
    draw_box(ax, "管理员模块", 4, 8.5, 2, 0.8)
    draw_box(ax, "干部模块", 2, 8.5, 2, 0.8)
    draw_box(ax, "用户模块", 6, 8.5, 2, 0.8)

    # 添加类
    draw_box(ax, "管理员类\n- 用户管理\n- 干部管理\n- 资源管理\n- 人口管理\n- 村务参与管理\n- 村务公开管理\n- 事务处理管理\n- 统计分析",
             4, 6, 2.5, 2)
    draw_box(ax, "干部类\n- 干部管理\n- 资源管理\n- 人口管理\n- 村务参与管理\n- 村务公开管理\n- 事务处理管理",
             2, 6, 2.5, 2)
    draw_box(ax, "用户类\n- 个人中心\n- 村务参与\n- 村务公开查看",
             6, 6, 2.5, 2)

    # 添加子功能
    draw_box(ax, "资源管理\n- 土地资源\n- 村民住房", 2, 3, 2.5, 1)
    draw_box(ax, "人口管理\n- 农户信息\n- 党员信息\n- 户籍信息", 5, 3, 2.5, 1)
    draw_box(ax, "村务公开管理\n- 政策通知\n- 财务收支", 8, 3, 2.5, 1)
    draw_box(ax, "村务参与管理\n- 投票选举\n- 意见反馈", 3, 1.5, 2.5, 1)
    draw_box(ax, "事务处理管理\n- 项目进展\n- 反馈处理", 6, 1.5, 2.5, 1)
    draw_box(ax, "统计分析\n- 用户活跃度分析\n- 村务管理效率分析", 9, 1.5, 2.5, 1)

    # 绘制箭头连接
    draw_arrow(ax, 5, 8.5, 5, 7.9)  # 管理员模块到管理员类
    draw_arrow(ax, 3, 8.5, 3, 7.9)  # 干部模块到干部类
    draw_arrow(ax, 7, 8.5, 7, 7.9)  # 用户模块到用户类

    draw_arrow(ax, 5, 6, 3.25, 4)  # 管理员类到资源管理
    draw_arrow(ax, 5, 6, 6.25, 4)  # 管理员类到人口管理
    draw_arrow(ax, 5, 6, 8.25, 4)  # 管理员类到村务公开管理
    draw_arrow(ax, 5, 6, 4, 2.5)   # 管理员类到村务参与管理
    draw_arrow(ax, 5, 6, 6.5, 2.5) # 管理员类到事务处理管理
    draw_arrow(ax, 5, 6, 9.25, 2.5) # 管理员类到统计分析

    plt.show()

# 运行主函数
main()
