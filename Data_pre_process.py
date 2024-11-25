# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 14:32:35 2023

@author: Wenjie Lan
"""
import pandas as pd

# 读取数据框 df_1 和 df_2
df_1 = pd.read_csv(r"C:\Users\czj\Desktop\标签.csv")
df_2 = pd.read_csv(r"C:\Users\czj\Desktop\新闻.csv")

# 将时间列转换为 datetime 类型
df_1['date'] = pd.to_datetime(df_1['date'])
df_2['date'] = pd.to_datetime(df_2['date'])

# 合并两个数据框，按照股票代码（stock_code）和时间（date）进行匹配
merged_df = pd.merge(df_1, df_2, on=['code', 'date'], how='inner')

# 保存为新的csv文件
# merged_df.to_csv(r'D:\深度学习项目\merged_dataset.csv')
# 检查整个数据框中是否有空值
# 检查某一列是否有空值
# if merged_df['code'].isnull().values.any():
#     print("列中存在空值。")
# else:
#     print("列中没有空值。")
# if merged_df['date'].isnull().values.any():
#     print("列中存在空值。")
# else:
#     print("列中没有空值。")
# if merged_df['news'].isnull().values.any():
#     print("列中存在空值。")
# else:
#     print("列中没有空值。")

#在原数据框上删除 'news' 列中包含空值的行
columns_to_check = ['code', 'date', 'news']
merged_df.dropna(subset=columns_to_check , inplace=True)

#删除除了标签和News内容两列外的其他列
columns_to_keep = ['is_rise', 'news']
df_cleaned = merged_df[columns_to_keep]
#删除清除完数据集上重复的新闻内容及所在行
df_final = df_cleaned.drop_duplicates(subset=['news'], keep='first')
df_final.to_csv(r'D:\深度学习项目\merged_dataset.csv')

#将文件划分成两个数据集
df_part1 = df_final.sample(frac=0.7, random_state=42)  # 将数据的70%随机选择为第一部分，剩余为
df_part2 = df_final.drop(df_part1.index)  # 剩下的数据作为第二部分
# 将两个部分分别保存为不同的CSV文件
df_part1.to_csv(r'D:\深度学习项目\part1.csv', index=False)
df_part2.to_csv(r'D:\深度学习项目\part2.csv', index=False)
