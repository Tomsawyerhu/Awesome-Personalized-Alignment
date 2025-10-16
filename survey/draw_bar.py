import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# 输入你的数据（来自你提供的表格）
data = {
    'Year': [2020, 2021, 2022, 2023, 2024, 2025],
    'Count': [1, 2, 4, 40, 67, 34]
}
df = pd.DataFrame(data)

# 排序（按年份顺序，虽然已经是了）
df = df.sort_values('Year')

# 创建柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Year'].astype(str), df['Count'], color='steelblue', edgecolor='black', alpha=0.8)

# 添加标题和标签
# plt.title('Number of Papers by Year', fontsize=20, fontweight='bold')
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Papers', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 在每个柱子上方显示数值
for bar, count in zip(bars, df['Count']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom', fontsize=12)

# 可选：添加网格线（y轴方向）
plt.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.7)

# 调整布局避免标签被截断
plt.tight_layout()

# 保存为 PDF 文件
pdf_filename = 'paper_count_by_year.pdf'
with PdfPages(pdf_filename) as pdf:
    pdf.savefig()  # 保存当前图像
print(f"✅ 柱状图已保存为 '{pdf_filename}'")

# 显示图像
plt.show()
