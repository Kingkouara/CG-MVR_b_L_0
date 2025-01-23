import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

def generate_synthetic_data(N, M, L_0, b):

    phi = np.random.uniform(0, 1, N)
    
    sorted_phi = sorted(phi, reverse=True)

    rank_dict = {value: rank for rank, value in enumerate(sorted_phi, start=1)}

    # 真の順位リストR_0を生成
    R_0 = np.array([rank_dict[x] for x in phi])

    # 投票者b_iから見る候補者a_jの能力値Φ'_ijを生成
    phi_prime = np.zeros((M, N))
    R = np.zeros((M, N), dtype=int)

    for i in range(M):
        for j in range(N):
            lower_bound = phi[j] - phi[j] * (1 - b)
            upper_bound = phi[j] + (1 - phi[j]) * (1 - b)
            phi_prime[i, j] = np.random.uniform(lower_bound, upper_bound)

        # 各投票者のランキングを作成
        ranked_indices = np.argsort(-phi_prime[i, :])  # 降順でソート
        # ranking_length = np.random.randint(L_0 - 0.2*L_0, L_0 + 0.2*L_0)  # ランキングの長さ
        ranking_length = L_0


        for k in range(ranking_length):
            R[i, ranked_indices[k]] = k + 1

    return phi, R_0, phi_prime, R

def compute_competition_matrix(R, N, M):
    # 遷移行列P^iを計算し、競争行列Aを生成
    A = np.zeros((N, N), dtype=int)

    for i in range(M):
        P_i = np.zeros((N, N), dtype=int)
        for s in range(N):
            for t in range(N):
                if R[i, s] > 0 and R[i, t] > 0:
                    P_i[s, t] = 1 if R[i, s] <= R[i, t] else 0
        A += P_i

    return A
#入力データの正方行列A生成終了

def calculate_rankings(A, N):
    # 出次数、入次数、および比率g_jを計算
    d_out = np.sum(A, axis=1)
    d_in = np.sum(A, axis=0)
    g = (d_out + 1) / (d_in + 1)

    g_tmp = g.copy()
    # 値を降順にソートし、順位を割り当てる
    sorted_values = sorted(g_tmp, reverse=True)

    # 順位を辞書にマッピング（値: 順位）
    rank_dict = {value: rank for rank, value in enumerate(sorted_values, start=1)}

    # 元の配列の各要素に対応する順位を取得
    ranks = [rank_dict[x] for x in g_tmp]

    return g, np.array(ranks)

def calculate_kendall_tau_distance(R_0, R_hat):
    if len(R_0) != len(R_hat):
        raise ValueError("The input rankings must have the same length.")
    
    distance = 0
    n = len(R_0)
    for i in range(n):
        for j in range(i + 1, n):
            if (R_0[i] < R_0[j] and R_hat[i] > R_hat[j]) or (R_0[i] > R_0[j] and R_hat[i] < R_hat[j]):
                distance += 1
    return distance


# パラメータの設定
N = 10  # 候補者数
M = 10  # 投票者数
L_0 = 10  # 各投票者のランキングの基本長さ
# b = 0  # 表示精度

b_values = [0.5]  # ノイズパラメータの候補
D_values = []

for b in b_values:
    # データ生成
    phi, R_0, phi_prime, R = generate_synthetic_data(N, M, L_0, b)

    # 競争行列Aの計算
    A = compute_competition_matrix(R, N, M)

    # 出次数、入次数、比率g_j、最終ランキングベクトルR_hatの計算
    g, R_hat = calculate_rankings(A, N)

    # Kendall Tau 距離の計算
    D = calculate_kendall_tau_distance(R_0 - 1, R_hat - 1)
    D_values.append(D)
    # print(f"b: {b}, g{g}, R{R_hat}")


# # 結果を保存または表示
# phi_df = pd.DataFrame(phi, columns=["True Ability"])
# R_0_df = pd.DataFrame(R_0, columns=["True Rank"])
# phi_prime_df = pd.DataFrame(phi_prime, columns=[f"Candidate_{j+1}" for j in range(N)])
# R_df = pd.DataFrame(R, columns=[f"Candidate_{j+1}" for j in range(N)])
# A_df = pd.DataFrame(A, columns=[f"Candidate_{j+1}" for j in range(N)])
# g_df = pd.DataFrame(g, columns=["g_j"])
# R_hat_df = pd.DataFrame(R_hat, columns=["Final Rank"])
# result_df = pd.DataFrame({"b": b_values, "D": D_values})



# # ファイルに保存
# phi_df.to_csv("true_ability.csv", index=False)#真の能力値(N×1)
# R_0_df.to_csv("true_rank.csv", index=False)#真のランクリスト(N×1)
# phi_prime_df.to_csv("displayed_ability.csv", index=False)#b_iから見たa_iの能力値(M×N)
# R_df.to_csv("rankings.csv", index=False)#ランキング行列(M×N)
# A_df.to_csv("competition_matrix.csv", index=False)#競争行列(N×N)
# g_df.to_csv("g_values.csv", index=False)#(M×1)
# R_hat_df.to_csv("final_rankings.csv", index=False)#最終的なランキング集約ベクトル
# result_df.to_csv("kendall_tau_distance_vs_b.csv", index=False)

# #グラフの描画
# plt.figure(figsize=(8, 6))
# plt.plot(result_df["b"], result_df["D"], marker='o', linestyle='-', label='Kendall Tau Distance')
# plt.title("Kendall Tau Distance vs b", fontsize=14)
# plt.xlabel("b (Accuracy Parameter)", fontsize=12)
# plt.ylabel("D (Kendall Tau Distance)", fontsize=12)
# plt.xticks(b_values)
# plt.legend()
# plt.grid()
# plt.savefig("kendall_tau_distance_vs_b.png")
# plt.show()


# # 結果を出力
# print("Kendall Tau 距離とbの関係がCSVファイルに保存されました: kendall_tau_distance_vs_b.csv")

