import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CG_method
import MVR_method
import time  # 追加: 時間計測用モジュールのインポート


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

if __name__ == '__main__':
    total_start_time = time.time()

    # パラメータの設定
    N = 50  # 候補者数
    M = 500  # 投票者数
    L_0 = 30  # 各投票者のランキングの基本長さ
    # b = 0  # 表示精度

    b_values = [0.95]  # ノイズパラメータの候補
    CG_D_values = []
    MVR_D_values = []  # 修正: 'vlaues' を 'values' に変更

    for b in b_values:
        # データ生成
        phi, R_0, phi_prime, R = generate_synthetic_data(N, M, L_0, b)

        # 競争行列Aの計算
        A = compute_competition_matrix(R, N, M)

        # 出次数、入次数、比率g_j、最終ランキングベクトルR_hatの計算
        g, R_hat = CG_method.calculate_rankings(A, N)

        # Kendall Tau 距離の計算
        # 一貫性を持たせるために R_0 と R_hat の処理を統一
        CG_D = CG_method.calculate_kendall_tau_distance(R_0 - 1, R_hat - 1)
        CG_D_values.append(CG_D)
        print("CG:ケンドールタウ距離D:")
        print(CG_D)
        final_solution = MVR_method.iterative_constraint_relaxation(A)
        # print("最終的なランキング行列X:")
        # print(final_solution)

        g_mvr, R_mvr = MVR_method.generate_final_ranking_vector(final_solution)
        # print("最終的なランキングベクトルR_mvr:")
        # print(R_mvr)

        # ケンドールタウ距離を計算
        MVR_D = MVR_method.calculate_kendall_tau_distance(R_mvr, R_0)
        MVR_D_values.append(MVR_D)  # 修正: 'MVR_D_vlaues' を 'MVR_D_values' に変更
        print("MVR:ケンドールタウ距離D:")
        print(MVR_D)


        # 全体の計算時間を計測
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"\nプログラム全体の計算時間: {total_elapsed_time:.4f} 秒")