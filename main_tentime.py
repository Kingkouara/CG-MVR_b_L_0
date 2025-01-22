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
        # ランキングの長さを整数に変換し、Nを超えないように制御
        ranking_length = int(np.random.randint(int(L_0 - 0.2*L_0), int(L_0 + 0.2*L_0)+1))
        
        for k in range(min(ranking_length, N)):  # Nを超えないようにする
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
    num_iterations = 10  # 繰り返し回数の設定
    b_value = 0.95  # 固定のノイズパラメータ
    N = 50  # 候補者数
    M = 500  # 投票者数
    L_0 = 30  # 各投票者のランキングの基本長さ

    # 結果を格納するリストの初期化
    CG_D_results = []
    MVR_D_results = []
    elapsed_times = []

    for iteration in range(1, num_iterations + 1):
        print(f"\n=== 繰り返し {iteration}/{num_iterations} を開始 ===")
        start_time = time.time()
        
        # データ生成
        phi, R_0, phi_prime, R = generate_synthetic_data(N, M, L_0, b_value)
        
        # 競争行列Aの計算
        A = compute_competition_matrix(R, N, M)
        
        # 出次数、入次数、比率g_j、最終ランキングベクトルR_hatの計算
        g, R_hat = CG_method.calculate_rankings(A, N)
        
        # Kendall Tau 距離の計算
        # 一貫性を持たせるために R_0 と R_hat の処理を統一
        CG_D = CG_method.calculate_kendall_tau_distance(R_0 - 1, R_hat - 1)
        CG_D_results.append(CG_D)
        print(f"繰り返し {iteration}: CG法のケンドールタウ距離D = {CG_D}")
        
        # MVR法の実行
        final_solution = MVR_method.iterative_constraint_relaxation(A)
        g_mvr, R_mvr = MVR_method.generate_final_ranking_vector(final_solution)
        
        # ケンドールタウ距離を計算
        MVR_D = MVR_method.calculate_kendall_tau_distance(R_mvr, R_0)
        MVR_D_results.append(MVR_D)
        print(f"繰り返し {iteration}: MVR法のケンドールタウ距離D = {MVR_D}")
        
        # 計算時間の記録
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)
        print(f"繰り返し {iteration} の計算時間: {elapsed_time:.4f} 秒")
    
    # 結果をデータフレームにまとめる
    results_df = pd.DataFrame({
        "Iteration": range(1, num_iterations + 1),
        "CG_D": CG_D_results,
        "MVR_D": MVR_D_results,
        "Elapsed_Time_sec": elapsed_times
    })
    

    # 結果の表示
    print("\n=== 全繰り返しの結果 ===")
    print(results_df)
    
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"\nプログラム全体の計算時間: {total_elapsed_time:.4f} 秒")

    # 結果をCSVファイルに保存
    results_df.to_csv("CG_and_MVR_kendall_tau_distance_results.csv", index=False)
    print("\n結果を 'kendall_tau_distance_results.csv' に保存しました。")
    

    # # # 結果を保存または表示
    # phi_df = pd.DataFrame(phi, columns=["True Ability"])
    # R_0_df = pd.DataFrame(R_0, columns=["True Rank"])
    # phi_prime_df = pd.DataFrame(phi_prime, columns=[f"Candidate_{j+1}" for j in range(N)])
    # R_df = pd.DataFrame(R, columns=[f"Candidate_{j+1}" for j in range(N)])
    # A_df = pd.DataFrame(A, columns=[f"Candidate_{j+1}" for j in range(N)])
    # g_df = pd.DataFrame(g, columns=["g_j"])
    # R_hat_df = pd.DataFrame(R_hat, columns=["Final Rank"])
    # result_df = pd.DataFrame({"b": b_values, "CG_D": CG_D_values, "MVR_D": MVR_D_values})  # 修正

    # # ファイルに保存
    # phi_df.to_csv("true_ability.csv", index=False)  # 真の能力値(N×1)
    # R_0_df.to_csv("true_rank.csv", index=False)  # 真のランクリスト(N×1)
    # phi_prime_df.to_csv("displayed_ability.csv", index=False)  # b_iから見たa_iの能力値(M×N)
    # R_df.to_csv("rankings.csv", index=False)  # ランキング行列(M×N)
    # A_df.to_csv("competition_matrix.csv", index=False)  # 競争行列(N×N)
    # g_df.to_csv("g_values.csv", index=False)  # (M×1)
    # R_hat_df.to_csv("final_rankings.csv", index=False)  # 最終的なランキング集約ベクトル
    # result_df.to_csv("kendall_tau_distance_vs_b.csv", index=False)

    # # グラフの描画
    # # CG法の結果
    # plt.plot(b_values, CG_D_values, label="CG method")
    # # MVR法の結果
    # plt.plot(b_values, MVR_D_values, label="MVR method")  # 修正: 'MVR_D_vlaues' を 'MVR_D_values' に変更
    # plt.xlabel("b")
    # plt.ylabel("Kendall Tau Distance")
    
    # plt.legend()
    # plt.title("Kendall Tau Distance vs b")
    # plt.grid(True)  # グリッドを追加して見やすくする（オプション）
    # plt.savefig("kendall_tau_distance_plot.png")  # プロットをファイルに保存（オプション）
    # # plt.show()  # プロットを表示

