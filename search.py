import jsonlines
import pandas as pd
from collections import Counter
import requests
from xml.etree import ElementTree as ET
import urllib.parse
import time

# ================== 配置 ==================
file_path = '/Users/tom/Downloads/personalized_papers.xlsx'  # 修改为你的路径
target_columns = ['paper', 'abbr', 'year', 'source', 'tag']

# 存储结果：按 category (sheet_name) 分组
results_by_category = {}

# arXiv API 查询函数
def get_arxiv_link(title, retries=2, delay=1):
    """
    根据论文标题查询 arXiv，返回第一篇匹配的结果链接
    """
    query = urllib.parse.quote_plus(title.strip())
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results=1"

    headers = {
        "User-Agent": "ResearchBot/1.0 (Academic Tool; https://example.com)"
    }

    for _ in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                time.sleep(delay)
                continue

            root = ET.fromstring(response.content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            entries = root.findall('atom:entry', ns)
            if not entries:
                return None, None

            entry = entries[0]
            paper_url = entry.find('atom:id', ns).text
            pdf_url = paper_url.replace("/abs/", "/pdf/") + ".pdf"
            return paper_url, pdf_url

        except Exception as e:
            print(f"  🔁 重试中... 错误: {e}")
            time.sleep(delay)
    return None, None


# ================== 主程序 ==================
with pd.ExcelFile(file_path) as xls:
    for sheet_name in xls.sheet_names:
        try:
            # 读取列名判断是否匹配
            df_header = pd.read_excel(xls, sheet_name=sheet_name, nrows=0)
            columns = df_header.columns.str.strip().tolist()

            if columns == target_columns:
                print(f"✅ 处理匹配的 sheet: {sheet_name}")
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df.columns = df.columns.str.strip()  # 清理列名空格

                # 初始化该 category 的列表
                category = sheet_name
                if category not in results_by_category:
                    results_by_category[category] = []

                # 遍历每一行
                for _, row in df.iterrows():
                    paper = row['paper']
                    abbr = row.get('abbr', '') or ''
                    year = row.get('year', '')
                    source = row.get('source', '')
                    tag = row.get('tag', '')

                    if pd.isna(paper) or not str(paper).strip():
                        continue  # 跳过空标题

                    print(f"🔍 查询: {paper}")

                    # 查询 arXiv
                    abs_url, pdf_url = get_arxiv_link(paper)

                    if abs_url:
                        status = "✅ 成功"
                    else:
                        abs_url = "#"
                        pdf_url = "#"
                        status = "❌ 未找到"

                    # 构建条目
                    item = {
                        "Title": f"[{paper}]({abs_url})",
                        "Abbr": abbr,
                        "Year": year,
                        "Source": source,
                        "Category": category,
                        "Tag": tag,
                        "PDF": f"[PDF]({pdf_url})" if pdf_url and pdf_url != "#" else "–",
                        "Status": status
                    }

                    # 添加到当前 category 组
                    results_by_category[category].append(item)

                    # 追加写入 JSONL（每条独立记录）
                    with jsonlines.open('./papers.jsonl', 'a') as f:
                        f.write(item)

                    # 控制请求频率
                    time.sleep(1.5)

        except Exception as e:
            print(f"⚠️ 读取 sheet {sheet_name} 出错: {e}")

# ================== 生成 Markdown 输出（按 category 分组）==================
if results_by_category:
    md_lines = []

    for category, items in results_by_category.items():
        if not items:
            continue

        # 添加大标题
        md_lines.append(f"# {category}")
        md_lines.append("")
        md_lines.append("| 论文标题 | 缩写 | 年份 | 来源 | 标签 | 下载 | 状态 |")
        md_lines.append("|----------|------|------|------|------|--------|--------|")

        for r in items:
            md_lines.append(
                f"| {r['Title']} | {r['Abbr']} | {r['Year']} | {r['Source']} | {r['Tag']} | {r['PDF']} | {r['Status']} |"
            )
        md_lines.append("")  # 空行分隔不同 category

    markdown_output = "\n".join(md_lines)

    # 打印到控制台
    print("\n" + "=" * 60)
    print(markdown_output)

    # 保存到文件
    with open("papers.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"\n📄 已保存为 papers.md （共 {sum(len(v) for v in results_by_category.values())} 篇论文）")

else:
    print("❌ 没有提取到任何论文数据。")

