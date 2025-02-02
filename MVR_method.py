import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog 

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


# ヒルサイド違反の数を表す行列Cを作成する関数
# 入力: A (競争行列)
# 出力: ヒルサイド違反の数を表す行列C
def calculate_C(A):
    n = A.shape[0]
    # sum_C = 0
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # C[i, j] = np.sum(A[i, :] < A[j, :]) + np.sum(A[:, i] > A[:, j])
                C[i, j] = A[i, j] - A[j, i]

    # sum_C = np.sum(C)
    # print("ヒルサイド違反行列Cの合計値:",sum_C)

    return C

# BILPを定式化してLPで緩和する関数
# 入力: C (ヒルサイド違反行列)
# 出力: 緩和されたLPの解xと制約リストA_ineq, b_ineq
def solve_MVR_LP(C):
    n = C.shape[0]
    num_variables = n * n

    # 目的関数: c_ij * x_ij の係数ベクトル
    c = C.flatten()
    c= -c

    # 制約: x_ij + x_ji = 1 (タイプ1: 反対称制約)
    A_eq = np.zeros((n * (n - 1) // 2, num_variables))  # 等式制約行列
    b_eq = np.ones(n * (n - 1) // 2)  # 等式制約の右辺値
    eq_idx = 0  # 等式制約の行インデックス
    for i in range(n):
        for j in range(i + 1, n):  # 上三角成分のペアを探索
            A_eq[eq_idx, i * n + j] = 1  # x_ij の係数を1に設定
            A_eq[eq_idx, j * n + i] = 1  # x_ji の係数を1に設定
            eq_idx += 1

   

    # 制約: 0 <= x_ij <= 1 (バイナリ制約の緩和)
    bounds = [(0, 1) for _ in range(num_variables)]

    # 初期LPを解く
    result = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')  # 定式化を解く関数
    if result.success:
        x = result.x.reshape((n, n))
        return x, A_eq, b_eq
    else:
        raise ValueError("初期LPの解が見つかりませんでした。")

# 制約緩和を用いた反復的なLP解法
# 入力: A (競争行列)
# 出力: 最終的なランキング行列 x
def iterative_constraint_relaxation(A):
    C = calculate_C(A)  # ヒルサイド違反行列を計算
    c = C.flatten()
    c = -c 
    n = C.shape[0]  # アイテム数
    x, A_eq, b_eq = solve_MVR_LP(C)  # 初期LPを解く

    # 制約: x_ij + x_jk + x_ki <= 2 (タイプ2: 伝播性制約、初期は緩和)
    A_ineq = []  # 不等式制約行列（動的に拡張）（A_eqのタイプ２ver）
    b_ineq = []  # 不等式制約の右辺ベクトル

    if x is None or np.all(x == 0):  # 初期解がすべてゼロの場合の対策
        raise ValueError("初期ランキング行列が不正です。解法を確認してください。")

    while True:
        violations = []  # 制約違反のリスト
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:  # i, j, k が異なる場合のみチェック
                        if x[i, j] + x[j, k] + x[k, i] > 2:  # 制約違反を検出
                            violations.append((i, j, k))

        # 違反がない場合は終了
        if not violations:
            break

        # 違反制約を追加
        for i, j, k in violations:
            # print(f"違反制約: {violations}")
            row = np.zeros(n * n)  # 新しい制約の係数行
            # print(f"新しい制約の係数行: {row}")
            row[i * n + j] = 1
            row[j * n + k] = 1
            row[k * n + i] = 1
            # print(f"新しい制約の係数行: {row}")
            A_ineq.append(row)  # 新たに制約した1行を追加
            b_ineq.append(2)  # 右辺値を設定

        # 制約行列を再確認
        A_ineq_matrix = np.array(A_ineq)
        b_ineq_vector = np.array(b_ineq)


        # 再実行
        bounds = [(0, 1) for _ in range(n*n)]

        result = opt.linprog(c, A_ub=A_ineq_matrix, b_ub=b_ineq_vector, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            x = result.x.reshape((n, n))
            if x is None or np.all(x == 0):  # xがすべてゼロの場合の対策
                print(f"x: {x}")
                raise ValueError("ランキング行列が不正です。解法を確認してください。")
        else:
            raise ValueError(f"再実行LPの解が見つかりませんでした: {result.message}")

    return x


# 最終的なランキングベクトルを生成する関数
# 入力: X (最終的なランキング行列)
# 出力: R_mvr (最終的なランキングベクトル)
def generate_final_ranking_vector(X):
    # print("最終的なランキング行列X:",X)
    # ①: 行の合計値を計算
    row_sums = np.sum(X, axis=1)

    tmp_sums = row_sums.copy()
    # 値を降順にソートし、順位を割り当てる
    sorted_values = sorted(tmp_sums, reverse=True)

    # 順位を辞書にマッピング（値: 順位）
    rank_dict = {value: rank for rank, value in enumerate(sorted_values, start=1)}

    # 元の配列の各要素に対応する順位を取得
    ranks = [rank_dict[x] for x in tmp_sums]

    return row_sums, np.array(ranks)
   

# ケンドールタウ距離を計算する関数
# 入力: R_mvr (推定されたランキング), R_0 (真のランキング)
# 出力: ケンドールタウ距離D
def calculate_kendall_tau_distance(R_0, R_mvr):
    if len(R_0) != len(R_mvr):
        raise ValueError("The input rankings must have the same length.")

    N = len(R_mvr)
    D = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (R_mvr[i] < R_mvr[j] and R_0[i] > R_0[j]) or (R_mvr[i] > R_mvr[j] and R_0[i] < R_0[j]):
                D += 1
    return D
