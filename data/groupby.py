from io import StringIO

import pandas as pd

# 读取csv文件
df = pd.read_csv('中国石化.csv', encoding='GBK', parse_dates=['日期'])

# 去掉小时分钟，只保留年月日
df['日期'] = pd.to_datetime(df['日期']).dt.date

# 按日期分组，聚合数据
df_merged = df.groupby(['日期']).agg({
    '标题': lambda x: ';'.join(x),
    # ... 可以根据实际情况添加需要合并的列
})

# 重置索引，添加新的日期列
df_merged = df_merged.reset_index()

# 保存为csv文件
df_merged.to_csv('merged_csv_file.csv', index=False)

