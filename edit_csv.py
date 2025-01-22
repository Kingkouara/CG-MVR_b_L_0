import pandas as pd

# CSVファイルを読み込む
def rearrange_csv_labels(input_csv, output_csv):
    # データを読み込み
    data = pd.read_csv(input_csv)

    # 元のラベルを取得
    labels = data.columns.tolist()

    if labels[0] != "Iteration":
        raise ValueError("The first label must be 'Iteration'")

    iteration_label = labels[0]
    cg_labels = [label for label in labels if label.startswith("CG_D")]
    mvr_labels = [label for label in labels if label.startswith("MVR_D")]

    # bの値でソート
    cg_sorted = sorted(cg_labels, key=lambda x: float(x.split("=")[1][:-1]))
    mvr_sorted = sorted(mvr_labels, key=lambda x: float(x.split("=")[1][:-1]))

    # 交互に並べ替え
    alternating_labels = [item for pair in zip(cg_sorted, mvr_sorted) for item in pair]
    new_labels = [iteration_label] + alternating_labels

    # データフレームの列を並べ替え
    new_data = data[new_labels]

    # 新しいCSVファイルを保存
    new_data.to_csv(output_csv, index=False)

# 使用例
input_csv = "CG_and_MVR_wide_format.csv"  # 入力CSVファイル名
output_csv = "output.csv"  # 出力CSVファイル名
rearrange_csv_labels(input_csv, output_csv)
