import numpy as np
import pandas as pd
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
        
        # ランダムにL_0人を選ぶ
        candidates = np.random.choice(N, size=L_0, replace=False)  # ランダムにL_0人を選ぶ
        
        # 選ばれた候補者集合でphi_primeを降順ソートしてランク付け
        selected_phi_prime = phi_prime[i, candidates]  # 選ばれた候補者のphi_prime値
        sorted_indices = np.argsort(-selected_phi_prime)  # 降順ソートのインデックス
        
        for rank, idx in enumerate(sorted_indices):
            R[i, candidates[idx]] = rank + 1  # 1から順位を付ける
    
    return phi, R_0, phi_prime, R

def compute_competition_matrixB(R, N, M):
    # 遷移行列P^iを計算し、競争行列Bを生成
    B = np.zeros((N, N), dtype=int)
    a = 2

    for i in range(M):
        P_i = np.zeros((N, N), dtype=int)
        alpha = 2

        for s in range(N):
            for t in range(N):
                if R[i, s] > 0 and R[i, t] > 0:
                    if R[i, s] < R[i, t]:
                            P_i[s, t] = 1  # sがtに勝った場合
                    elif R[i, s] > R[i, t]:
                            P_i[s, t] = alpha  # sがtに負けた場合
       
        B += P_i

    return B
#入力データの正方行列B生成終了

def compute_markov_ranking(B):
    N = B.shape[0]
    
    # 行正規化して遷移行列を作成
    row_sums = B.sum(axis=1)
    P = B / row_sums[:, np.newaxis]

    # 固有ベクトルを求める (ステディステート)
    eigvals, eigvecs = np.linalg.eig(P.T)
    steady_state = eigvecs[:, np.isclose(eigvals, 1)]
    steady_state = steady_state[:, 0].real # 実数部分のみ取得
    steady_state = steady_state / steady_state.sum()  # 正規化

    steady_state_tmp = steady_state.copy()

    # 値を降順にソートし、順位を割り当てる
    sorted_values = sorted(steady_state_tmp, reverse=True)

    # 順位を辞書にマッピング（値: 順位）
    rank_dict = {value: rank for rank, value in enumerate(sorted_values, start=1)}

    # 元の配列の各要素に対応する順位を取得
    ranks = [rank_dict[x] for x in steady_state_tmp]

    return steady_state, np.array(ranks)

def calculate_kendall_tau_distance(R_0, R_alpha):
    if len(R_0) != len(R_alpha):
        raise ValueError("The input rankings must have the same length.")
    
    distance = 0
    n = len(R_0)
    for i in range(n):
        for j in range(i + 1, n):
            if (R_0[i] < R_0[j] and R_alpha[i] > R_alpha[j]) or (R_0[i] > R_0[j] and R_alpha[i] < R_alpha[j]):
                distance += 1
    
    # print(f"D_alpha: {distance}")
    return distance

# # パラメータの設定
# N = 50  # 候補者数
# M = 500  # 投票者数
# L_0 = 30  # 各投票者のランキングの基本長さ
# # b_values = [0.5]  # 表示精度
# b_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
# D_values = []

# for b in b_values:
#     # データ生成
#     phi, R_0, phi_prime, R = generate_synthetic_data(N, M, L_0, b)

#     # 競争行列Bの計算
#     B = compute_competition_matrixB(R, N, M)

#     #固有ベクトル、最終ランキングベクトルR_alphaの計算
#     steady_state, R_alpha = compute_markov_ranking(B)

#     # Kendall Tau 距離の計算
#     D = calculate_kendall_tau_distance(R_0, R_alpha)
#     D_values.append(D)

# # 結果を保存または表示
# phi_df = pd.DataFrame(phi, columns=["True Ability"])
# R_0_df = pd.DataFrame(R_0, columns=["True Rank"])
# phi_prime_df = pd.DataFrame(phi_prime, columns=[f"Candidate_{j+1}" for j in range(N)])
# R_df = pd.DataFrame(R, columns=[f"Candidate_{j+1}" for j in range(N)])
# B_df = pd.DataFrame(B, columns=[f"Candidate_{j+1}" for j in range(N)])
# steady_state_df = pd.DataFrame(steady_state, columns=["steady_state_j"])
# R_alpha_df = pd.DataFrame(R_alpha, columns=["Final Rank"])
# result_df = pd.DataFrame({"b": b_values, "D": D_values})



# # ファイルに保存
# phi_df.to_csv("true_ability.csv", index=False)#真の能力値(N×1)
# R_0_df.to_csv("true_rank.csv", index=False)#真のランクリスト(N×1)
# phi_prime_df.to_csv("displayed_ability.csv", index=False)#b_iから見たa_iの能力値(M×N)
# R_df.to_csv("rankings.csv", index=False)#ランキング行列(M×N)
# B_df.to_csv("competition_matrixB.csv", index=False)#競争行列(N×N)
# steady_state_df.to_csv("steady_state_values.csv", index=False)#(N×1)
# R_alpha_df.to_csv("final_rankings.csv", index=False)#最終的なランキング集約ベクトル
# result_df.to_csv("kendall_tau_distance_vs_b.csv", index=False)

