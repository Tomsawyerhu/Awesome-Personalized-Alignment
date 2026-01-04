import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# 输入你的数据
data = {
    'Year': [2020, 2021, 2022, 2023, 2024, 2025],
    'Count': [1, 2, 4, 40, 67, 34]
}
df = pd.DataFrame(data)
df = df.sort_values('Year')

# 生成蓝色渐变
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))  # 0.4-0.9区间为较浅到较深蓝

plt.figure(figsize=(10, 6))
bars = plt.bar(
    df['Year'].astype(str),
    df['Count'],
    color=colors,
    edgecolor='black',
    alpha=0.5  # 透明度0.6
)

# 添加标签
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Papers', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 在每个柱子上方显示数值
for bar, count in zip(bars, df['Count']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom', fontsize=12)

# 不显示网格线
plt.grid(False)

plt.tight_layout()

# 保存为 PDF 文件
pdf_filename = 'paper_count_by_year.pdf'
with PdfPages(pdf_filename) as pdf:
    pdf.savefig()
print(f"✅ 柱状图已保存为 '{pdf_filename}'")

plt.show()
