import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

# CSVファイルのパス
csv_files = glob.glob("*.csv")

plt.figure(figsize=(10, 6))

# 各CSVファイルを読み込み、プロット
for csv_file in csv_files:
    # データの読み込み
    data = pd.read_csv(csv_file)

    # bの値を取得
    b_values = data.columns[1:].astype(float)

    # 各bの値ごとに平均を計算
    averages = data.iloc[:, 1:].mean(axis=0)

    # プロット
    plt.plot(b_values, averages,marker='o', label=csv_file)

# グラフの装飾
plt.title('Average Values by b (Multiple Files)', fontsize=14)
plt.xlabel('b values', fontsize=12)
plt.ylabel('Average', fontsize=12)
plt.legend(title="CSV Files")
plt.grid(True)
plt.show()
