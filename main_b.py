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
        
        # 降順ソートでランキング作成
        ranked_indices = np.argsort(-phi_prime[i, :])
        ranking_length = int(np.random.randint(
            int(L_0 - 0.2 * L_0), 
            int(L_0 + 0.2 * L_0) + 1
        ))
        
        for k in range(min(ranking_length, N)):
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
    # 計測開始
    total_start_time = time.time()
    
    # --------------------------
    # ここを変える
    num_iterations = 10  
    b_values = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    # --------------------------
    
    # 固定パラメータ
    N = 50
    M = 500
    L_0 = 25  # ここを変える
    
    # 全試行結果を入れるリスト
    all_results = []
    
    # b 値ごとにループ
    
    for b in b_values:
        print(f"\n=== ノイズパラメータ b = {b} の処理を開始 ===")
        
        # iteration を回す
        for iteration in range(1, num_iterations + 1):
            print(f"  - Iteration {iteration}/{num_iterations}...")
            
            start_time = time.time()
            
            # データ生成
            phi, R_0, phi_prime, R = generate_synthetic_data(N, M, L_0, b)
            
            # 競争行列Aの計算
            A = compute_competition_matrix(R, N, M)
            
            # CG法
            g, R_hat = CG_method.calculate_rankings(A, N)
            CG_D = CG_method.calculate_kendall_tau_distance(R_0 - 1, R_hat - 1)
            
            # MVR法
            final_solution = MVR_method.iterative_constraint_relaxation(A)
            g_mvr, R_mvr = MVR_method.generate_final_ranking_vector(final_solution)
            MVR_D = MVR_method.calculate_kendall_tau_distance(R_mvr, R_0)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"    [CG_D: {CG_D:.4f}, MVR_D: {MVR_D:.4f}, time: {elapsed_time:.2f}s]")
            
            # （縦持ち形式で）結果を追加
            all_results.append({
                "b_value": b,
                "Iteration": iteration,
                "CG_D": CG_D,
                "MVR_D": MVR_D
            })
    
    # 全処理終了
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"\n=== 全処理完了: 総計算時間 {total_elapsed_time:.2f} 秒 ===\n")

    # DataFrameに変換
    results_df = pd.DataFrame(all_results)
    print("▼ 縦持ち形式の DataFrame")
    print(results_df.head(15))  # 確認用
    
    # ----
    # ここから CG_D のみを横持ちにする
    # 行: Iteration (1..10)
    # 列: b_value (0.91..0.99)
    # => 10行×9列の形になる
    # ----
    pivoted_cg = results_df.pivot(
        index="Iteration", 
        columns="b_value", 
        values="CG_D"  # ← CG_Dのみ
    )
    
    print("\n▼ CG_D を横持ちにしたピボットテーブル (10×9 の形)")
    print(pivoted_cg)
    
    # CSV出力
    pivoted_cg.to_csv(f"CG_D_wide_format_L_0.csv", float_format="%.4f")
    print("\n結果を 'CG_D_wide_format.csv' に保存しました。")

    # もし MVR_D も同様に 10×9 で欲しければ
    pivoted_mvr = results_df.pivot(
        index="Iteration",
        columns="b_value",
        values="MVR_D"
    )
    pivoted_mvr.to_csv(f"MVR_D_wide_format_L_0.csv", float_format="%.4f")
    print("結果を 'MVR_D_wide_format.csv' に保存しました。")

