import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CG_method
import alpha_method
import time  # 計算時間測定用

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


def compute_competition_matrix(R, N, M):
    # 遷移行列P^iを計算し、競争行列Aを生成
    A = np.zeros((N, N), dtype=int)
    for i in range(M):
        P_i = np.zeros((N, N), dtype=int)
        for s in range(N):
            for t in range(N):
                if R[i, s] > 0 and R[i, t] > 0:
                    P_i[s, t] = 1 if R[i, s] < R[i, t] else 0

        # print(f"P_{i}: {P_i}")
        A += P_i
    return A


def compute_competition_matrixB(R, N, M):
    # 遷移行列P^iを計算し、競争行列Bを生成
    B = np.zeros((N, N), dtype=int)

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

if __name__ == '__main__':
    # 計測開始
    total_start_time = time.time()
    
    # --------------------------
    # ここを変える
    num_iterations = 10
    b_values = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    # b_values = [0.5]
    # -------------------------
    
    # 固定パラメータ
    N = 50
    M = 500
    L_0 = [40]  # ここを変える

    # b 値ごとにループ
    for l_0 in L_0:
        # 全試行結果を入れるリスト
        all_results = []

        print(f"\n=== L_0 = {l_0} の処理を開始 ===")

        for b in b_values:
            print(f"\n=== ノイズパラメータ b = {b} の処理を開始 ===")

            # iteration を回す
            for iteration in range(1, num_iterations + 1):
                print(f"  - Iteration {iteration}/{num_iterations}...")

                start_time = time.time()

                # データ生成
                phi, R_0, phi_prime, R = generate_synthetic_data(N, M, l_0, b)

                # 競争行列A、Bの計算
                A = compute_competition_matrix(R, N, M)
                B = compute_competition_matrixB(R, N, M)

                # CG法
                g, R_hat = CG_method.calculate_rankings(A)
                CG_D = CG_method.calculate_kendall_tau_distance(R_0 - 1, R_hat - 1)

                #1,alpha法
                steady_state, R_alpha = alpha_method.compute_markov_ranking(B)
                alpha_D = alpha_method.calculate_kendall_tau_distance(R_0 - 1, R_alpha - 1)


                end_time = time.time()
                elapsed_time = end_time - start_time
                # print(f"[R_0: {R_0}]")
                # print(f"[R_hat: {R_hat}]")
                # print(f"[R_mvr: {R_mvr}]")
                print(f"    [CG_D: {CG_D:.4f}, alpha_D: {alpha_D:.4f}, time: {elapsed_time:.2f}s]")

                # （縦持ち形式で）結果を追加
                all_results.append({
                    "b_value": b,
                    "Iteration": iteration,
                    "CG_D": CG_D,
                    "MVR_D": alpha_D
                })

        # DataFrameに変換
        results_df = pd.DataFrame(all_results)
        # print("\n▼ 縦持ち形式の DataFrame")
        # print(results_df.head(15))  # 確認用

        # ----
        # CG_D のみを横持ちにする
        pivoted_cg = results_df.pivot(
            index="Iteration", 
            columns="b_value", 
            values="CG_D"  # ← CG_Dのみ
        )

        # print("\n▼ CG_D を横持ちにしたピボットテーブル")
        # print(pivoted_cg)

        # CSV出力
        # pivoted_cg.to_csv(f"CG_D_wide_format_L_0_{l_0}.csv", float_format="%.4f")
        # print(f"\n結果を 'CG_D_wide_format_L_0_{l_0}.csv' に保存しました。")
        pivoted_cg.to_csv(f"CG_D_L_0_{l_0}.csv", float_format="%.4f")
        print(f"\n結果を 'CG_D_L_0_{l_0}.csv' に保存しました。")

        # MVR_D も同様に処理
        pivoted_mvr = results_df.pivot(
            index="Iteration",
            columns="b_value",
            values="MVR_D"
        )
        # pivoted_mvr.to_csv(f"MVR_D_wide_format_L_0_{l_0}.csv", float_format="%.4f")
        # print(f"結果を 'MVR_D_wide_format_L_0_{l_0}.csv' に保存しました。")

        pivoted_mvr.to_csv(f"alpha_D_L_0_{l_0}.csv", float_format="%.4f")
        print(f"\n結果を 'alpha_D_L_0_{l_0}.csv' に保存しました。")

        phi_df = pd.DataFrame(phi, columns=["True Ability"])
        R_0_df = pd.DataFrame(R_0, columns=["True Rank"])
        phi_prime_df = pd.DataFrame(phi_prime, columns=[f"Candidate_{j+1}" for j in range(N)])
        R_df = pd.DataFrame(R, columns=[f"Candidate_{j+1}" for j in range(N)])
        A_df = pd.DataFrame(A, columns=[f"Candidate_{j+1}" for j in range(N)])
        B_df = pd.DataFrame(B, columns=[f"Candidate_{j+1}" for j in range(N)])
        g_df = pd.DataFrame(g, columns=["g_j"])
        R_hat_df = pd.DataFrame(R_hat, columns=["Final Rank"])
        R_alpha_df = pd.DataFrame(R_alpha, columns=["Final Rank"])
        
        phi_df.to_csv("true_ability.csv", index=False)#真の能力値(N×1)
        R_0_df.to_csv("true_rank.csv", index=False)#真のランクリスト(N×1)
        phi_prime_df.to_csv("displayed_ability.csv", index=False)#b_iから見たa_iの能力値(M×N)
        R_df.to_csv("rankings.csv", index=False)#ランキング行列(M×N)
        A_df.to_csv("competition_matrix.csv", index=False)#競争行列(N×N)
        B_df.to_csv("competition_matrixB.csv", index=False)
        R_hat_df.to_csv("CG_final_rankings.csv", index=False)#最終的なランキング集約ベクトル
        R_alpha_df.to_csv("alpha_final_rankings.csv", index=False)#最終的なランキング集約ベクトル