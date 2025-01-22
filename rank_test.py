import numpy as np

# 元の配列
values = [1.01324503, 1.05882353, 0.98095238, 0.62121212, 1.15267176]

# 値を降順にソートし、順位を割り当てる
sorted_values = sorted(values, reverse=True)

# 順位を辞書にマッピング（値: 順位）
rank_dict = {value: rank for rank, value in enumerate(sorted_values, start=1)}

# 元の配列の各要素に対応する順位を取得
ranks = [rank_dict[x] for x in values]

print(values)
print(sorted_values)
print(ranks)
