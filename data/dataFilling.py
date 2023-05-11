import pandas as pd
from dateutil.rrule import rrule, DAILY
from datetime import datetime

# 读入表格，假设日期列名为'date'
df = pd.read_csv('merged_csv_file.csv', encoding='GBK', parse_dates=['日期'], index_col='日期')

# 使用fillna()函数填充缺失值
df_filled = df.fillna(method='ffill')  # 或者使用bfill()向后填充

# 获取最早日期和最晚日期
start_date = df_filled.index.min()
end_date = df_filled.index.max()

# 生成完整日期序列并转换为DatetimeIndex对象
all_dates = pd.DatetimeIndex(list(rrule(freq=DAILY, dtstart=start_date, until=end_date)))

# 将序列作为索引，然后使用reindex()函数对数据框进行重采样
new_df = df_filled.reindex(all_dates, method='ffill')

# 保存填充后的数据框
new_df.to_csv('table_filled.csv')
