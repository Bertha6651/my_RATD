import pandas as pd

# 读取CSV文件
csv_file_path = '/Users/lzd/code/RATD_copy/TCN/TCN/word_cnn/data/electricity/electricity_2012_hour.csv'
df = pd.read_csv(csv_file_path)

# 分割数据集
train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1

train_df = df.iloc[:int(len(df)*train_ratio)]
valid_df = df.iloc[int(len(df)*train_ratio):int(len(df)*(train_ratio+valid_ratio))]
test_df = df.iloc[int(len(df)*(train_ratio+valid_ratio)):]

# 将DataFrame保存为文本文件
train_txt_path = '/Users/lzd/code/RATD_copy/TCN/TCN/word_cnn/data/electricity/train.txt'
valid_txt_path = '/Users/lzd/code/RATD_copy/TCN/TCN/word_cnn/data/electricity/valid.txt'
test_txt_path = '/Users/lzd/code/RATD_copy/TCN/TCN/word_cnn/data/electricity/test.txt'

train_df.to_csv(train_txt_path, index=False, header=None)
valid_df.to_csv(valid_txt_path, index=False, header=None)
test_df.to_csv(test_txt_path, index=False, header=None)

print(f"Data split completed and saved to:\n{train_txt_path}\n{valid_txt_path}\n{test_txt_path}")