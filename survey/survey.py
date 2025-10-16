import pandas as pd
from collections import Counter

# 输入文件路径
file_path = '/Users/tom/Downloads/personalizedsurvey.xlsx'  # 替换为你的文件路径

# 目标列头（第一行应完全匹配）
target_columns = ['paper', 'abbr', 'year', 'source', 'tag']

# 存储符合条件的数据
all_years = []
all_sources = []

# 使用 ExcelFile 获取所有 sheet 名称
with pd.ExcelFile(file_path) as xls:
    for sheet_name in xls.sheet_names:
        try:
            # 只读取第一行来判断列名
            df_header = pd.read_excel(xls, sheet_name=sheet_name, nrows=0)  # 不读数据，只读列名
            columns = df_header.columns.str.strip().tolist()  # 去除空格并转为列表

            if columns == target_columns:
                print(f"✅ 匹配的 sheet: {sheet_name}")

                # 重新读取整个 sheet 的数据
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df.columns = df.columns.str.strip()  # 清理列名空格

                # 提取 year 和 source 列
                all_years.extend(df['year'].dropna().astype(int).tolist())
                all_sources.extend(df['source'].dropna().tolist())

        except Exception as e:
            print(f"⚠️ 读取 sheet {sheet_name} 时出错: {e}")

# 统计分布
year_count = Counter(all_years)
source_count = Counter(all_sources)

# 转为 DataFrame 更清晰展示
year_dist = pd.DataFrame(year_count.items(), columns=['Year', 'Count']).sort_values('Year')
source_dist = pd.DataFrame(source_count.items(), columns=['Source', 'Count']).sort_values('Count', ascending=False)

# 输出结果
print("\n📊 Year 分布:")
print(year_dist)

print("\n🌐 Source 分布:")
print(source_dist)

