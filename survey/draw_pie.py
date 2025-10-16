import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 原始数据
data = {
    'Source': [
        'arxiv', 'SIGIR', 'WWW', 'NIPS', 'TOIS', 'EMNLP', 'CHI', 'ICLR', 'PMLR',
        'AAAI', 'COLING', 'JIII', 'NAACL', 'AISTATS', 'IPM', 'ICAART', 'WACV',
        'RecSys', 'UIST', 'WSDM', 'KDD', 'CVPR', 'DSE', 'AIED'
    ],
    'Count': [
        95, 10, 6, 5, 5, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    ]
}
df = pd.DataFrame(data)

# 数据处理：将 Count <= 1 的归为 Others
threshold = 2
large = df[df['Count'] > threshold]
others_count = df[df['Count'] <= threshold]['Count'].sum()
if others_count > 0:
    others_row = pd.DataFrame([{'Source': 'Others', 'Count': others_count}])
    final_df = pd.concat([large, others_row], ignore_index=True)
else:
    final_df = large

print("📊 最终用于绘图的数据：")
print(final_df)

# 开始画图
plt.figure(figsize=(8, 8))

# 颜色可选（使用 colormap 更美观）
colors = plt.cm.Set3(range(len(final_df)))

# 或者用 tight_layout(pad=0)
plt.tight_layout(pad=0)

# 创建饼图
wedges, texts, autotexts = plt.pie(
    final_df['Count'],
    labels=final_df['Source'],
    autopct=lambda pct: f'{pct:.1f}%',  # 同时显示百分比 + 实际数量
    colors=colors,
    startangle=90,
    textprops={'fontsize': 12},
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
)

# 设置标题
# plt.title('Distribution of Paper Sources\n(with sources ≤1 grouped as "Others")',
#           fontsize=16, fontweight='bold', pad=20)

# 确保圆形
plt.axis('equal')

# 调整布局
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# 或者用 tight_layout(pad=0)
plt.tight_layout()

# 保存为 PDF
pdf_filename = 'source_distribution_pie.pdf'
with PdfPages(pdf_filename) as pdf:
    pdf.savefig()
print(f"✅ 饼图已保存为 '{pdf_filename}'")

# 显示图像
plt.show()
