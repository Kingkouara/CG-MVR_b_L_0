import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import CG_method
import MVR_method
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
        candidates = np.random.choice(N, size=L_0, replace=False)
        
        # 選ばれた候補者集合でphi_primeを降順ソートしてランク付け
        selected_phi_prime = phi_prime[i, candidates]
        sorted_indices = np.argsort(-selected_phi_prime)
        
        for rank, idx in enumerate(sorted_indices):
            R[i, candidates[idx]] = rank + 1
    
    return phi, R_0, phi_prime, R


def compute_competition_matrix(R, N, M):
    A = np.zeros((N, N), dtype=int)
    for i in range(M):
        P_i = np.zeros((N, N), dtype=int)
        for s in range(N):
            for t in range(N):
                if R[i, s] > 0 and R[i, t] > 0:
                    P_i[s, t] = 1 if R[i, s] < R[i, t] else 0
        A += P_i
    return A


if __name__ == '__main__':
    total_start_time = time.time()
    
    num_iterations = 100
    b_values = [0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
   
    N = 50
    M = 500
    L_0 = [25, 30, 35, 40, 45, 50]

    for l_0 in L_0:
        all_results = []
        win_counts = {b: 0 for b in b_values}

        print(f"\n=== L_0 = {l_0} の処理を開始 ===")

        for b in b_values:
            print(f"\n=== ノイズパラメータ b = {b} の処理を開始 ===")

            for iteration in range(1, num_iterations + 1):
                print(f"  - Iteration {iteration}/{num_iterations}...")

                start_time = time.time()
                phi, R_0, phi_prime, R = generate_synthetic_data(N, M, l_0, b)
                A = compute_competition_matrix(R, N, M)

                g, R_hat = CG_method.calculate_rankings(A)
                CG_D = CG_method.calculate_kendall_tau_distance(R_0 - 1, R_hat - 1)

                final_solution = MVR_method.iterative_constraint_relaxation(A)
                g_mvr, R_mvr = MVR_method.generate_final_ranking_vector(final_solution)
                MVR_D = MVR_method.calculate_kendall_tau_distance(R_0, R_mvr)

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"    [CG_D: {CG_D:.4f}, MVR_D: {MVR_D:.4f}, time: {elapsed_time:.2f}s]")

                all_results.append({
                    "b_value": b,
                    "Iteration": iteration,
                    "CG_D": CG_D,
                    "MVR_D": MVR_D
                })

                if MVR_D < CG_D:
                    win_counts[b] += 1

        results_df = pd.DataFrame(all_results)
        pivoted_cg = results_df.pivot(index="Iteration", columns="b_value", values="CG_D")
        pivoted_cg.to_csv(f"CG_D_L_0_{l_0}.csv", float_format="%.4f")
        print(f"\n結果を 'CG_D_L_0_{l_0}.csv' に保存しました。")

        pivoted_mvr = results_df.pivot(index="Iteration", columns="b_value", values="MVR_D")
        pivoted_mvr.to_csv(f"MVR_D_L_0_{l_0}.csv", float_format="%.4f")
        print(f"\n結果を 'MVR_D_L_0_{l_0}.csv' に保存しました。")

        win_counts_df = pd.DataFrame(list(win_counts.items()), columns=["b_value", "MVR_win_count"])
        win_counts_df.to_csv(f"MVR_win_counts_L_0_{l_0}.csv", index=False)
        print(f"\nMVRが勝った回数を 'MVR_win_counts_L_0_{l_0}.csv' に保存しました。")

        phi_df = pd.DataFrame(phi, columns=["True Ability"])
        R_0_df = pd.DataFrame(R_0, columns=["True Rank"])
        phi_prime_df = pd.DataFrame(phi_prime, columns=[f"Candidate_{j+1}" for j in range(N)])
        R_df = pd.DataFrame(R, columns=[f"Candidate_{j+1}" for j in range(N)])
        A_df = pd.DataFrame(A, columns=[f"Candidate_{j+1}" for j in range(N)])
        g_df = pd.DataFrame(g, columns=["g_j"])
        R_hat_df = pd.DataFrame(R_hat, columns=["Final Rank"])
        R_mvr_df = pd.DataFrame(R_mvr, columns=["Final Rank"])
        
        phi_df.to_csv("true_ability.csv", index=False)
        R_0_df.to_csv("true_rank.csv", index=False)
        phi_prime_df.to_csv("displayed_ability.csv", index=False)
        R_df.to_csv("rankings.csv", index=False)
        A_df.to_csv("competition_matrix.csv", index=False)
        R_hat_df.to_csv("CG_final_rankings.csv", index=False)
        R_mvr_df.to_csv("MVR_final_rankings.csv", index=False)