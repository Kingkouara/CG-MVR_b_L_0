import numpy as np
from scipy.optimize import linprog

def calculate_C(A):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = np.sum(A[i, :] < A[j, :]) + np.sum(A[:, i] > A[:, j])
    return C

def solve_MVR_LP(C):
    n = C.shape[0]
    num_variables = n * n
    C_norm = C / np.max(C)
    c = C_norm.flatten()
    
    A_eq = np.zeros((n * (n - 1) // 2, num_variables))
    b_eq = np.ones(n * (n - 1) // 2)
    eq_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_eq[eq_idx, i * n + j] = 1
            A_eq[eq_idx, j * n + i] = 1
            eq_idx += 1
    
    bounds = [(0, 1) for _ in range(num_variables)]
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')
    if result.success:
        x = result.x.reshape((n, n))
        print("初期LPの解 X:")
        print(x)
        return x, A_eq, b_eq
    else:
        raise ValueError("初期LPの解が見つかりませんでした。")

def iterative_constraint_relaxation(C):
    C_norm = C / np.max(C)
    c = C_norm.flatten()
    n = C.shape[0]
    x, A_eq, b_eq = solve_MVR_LP(C)
    
    A_ineq = []
    b_ineq = []
    while True:
        violations = []
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        if x[i, j] + x[j, k] + x[k, i] > 2:
                            violations.append((i, j, k))
        
        if not violations:
            return x
        
        for i, j, k in violations:
            row = np.zeros(n * n)
            row[i * n + j] = 1
            row[j * n + k] = 1
            row[k * n + i] = 1
            A_ineq.append(row)
            b_ineq.append(2)
        
        A_ineq_matrix = np.array(A_ineq)
        b_ineq_vector = np.array(b_ineq)
        bounds = [(0, 1) for _ in range(n * n)]
        
        result = linprog(c, A_ub=A_ineq_matrix, b_ub=b_ineq_vector, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            x = result.x.reshape((n, n))
            print("更新後のランキング行列 X:")
            print(x)
        else:
            raise ValueError("再実行LPの解が見つかりませんでした。")
    
    return x

# Example usage
C = np.array([[0, 4, 2], 
              [2, 0, 2], 
              [2, 2, 0]])
ranking_matrix = iterative_constraint_relaxation(C)
print("最終的なランキング行列 X:")
print(ranking_matrix)
# import numpy as np

# def generate_violation_matrix(A):
#     """
#     Generate a violation matrix for a given 3x3 competition matrix A.
    
#     Parameters:
#     A (numpy.ndarray): 3x3 competition matrix.
    
#     Returns:
#     numpy.ndarray: 3x3 violation matrix.
#     """
#     n = A.shape[0]
#     C = np.zeros((n, n), dtype=int)
    
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 C[i, j] = np.sum(A[i, :] < A[j, :]) + np.sum(A[:, i] > A[:, j])
    
#     return C

# # Example usage
# A = np.array([[0, 1, 1],
#               [3, 0, 3],
#               [3, 1, 0]])

# violation_matrix = generate_violation_matrix(A)
# print("Competition Matrix A:")
# print(A)
# print("Violation Matrix C:")
# print(violation_matrix)


#以下、Φの得点差でやるやつ
import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def generate_synthetic_data(N, M, L_0, b):
    phi = np.random.uniform(0, 1, N)
    sorted_phi = sorted(phi, reverse=True)
    rank_dict = {value: rank for rank, value in enumerate(sorted_phi, start=1)}
    
    R_0 = np.array([rank_dict[x] for x in phi])
    phi_prime = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            lower_bound = phi[j] - phi[j] * (1 - b)
            upper_bound = phi[j] + (1 - phi[j]) * (1 - b)
            phi_prime[i, j] = np.random.uniform(lower_bound, upper_bound)
    
    return phi, R_0, phi_prime

def compute_score_difference_matrix(phi_prime,N):
    A= np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = np.mean(phi_prime[:, i] - phi_prime[:, j])
                A[i, j] = max(0, diff)
    
    return A

def calculate_C(A):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = np.sum(A[i, :] < A[j, :]) + np.sum(A[:, i] > A[:, j])
    return C

def solve_MVR_LP(C):
    n = C.shape[0]
    num_variables = n * n

    # 目的関数: c_ij * x_ij の係数ベクトル
    C_norm = C/np.max(C)  # 正規化
    c = C_norm.flatten()

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
    # bounds = [(0, 1) for _ in range(num_variables)]
    bounds = [(0, 1) for _ in range(num_variables)]

    # 初期LPを解く
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')  # 定式化を解く関数
    if result.success:
        x = result.x.reshape((n, n))

        # print("初期LPの解が見つかりました。",x)
        return x, A_eq, b_eq
    else:
        raise ValueError("初期LPの解が見つかりませんでした。")

# 制約緩和を用いた反復的なLP解法
# 入力: A (競争行列)
# 出力: 最終的なランキング行列 x
def iterative_constraint_relaxation(A):
    C = calculate_C(A)  # ヒルサイド違反行列を計算
    C_norm = C/np.max(C)  # 正規化
    c = C_norm.flatten()

    n = C.shape[0]  # アイテム数
    x, A_eq, b_eq = solve_MVR_LP(C)  # 初期LPを解く

    # 制約: x_ij + x_jk + x_ki <= 2 (タイプ2: 伝播性制約、初期は緩和)
    A_ineq = []  # 不等式制約行列（動的に拡張）（A_eqのタイプ２ver）
    b_ineq = []  # 不等式制約の右辺ベクトル

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
            # print("最終的なランキング行列X",x)
            # print("制約2",A_ineq)
            break

        # 違反制約を追加
        for i, j, k in violations:
            # print(f"違反制約の数:{len(violations)}")
            
            row = np.zeros(n * n)  # 新しい制約の係数行
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

        result = linprog(c, A_ub=A_ineq_matrix, b_ub=b_ineq_vector, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if result.success:
            x = result.x.reshape((n, n))
            if x is None or np.all(x == 0):  # xがすべてゼロの場合の対策
                # print(f"x: {x}")
                raise ValueError("ランキング行列が不正です。解法を確認してください。")
        else:
            raise ValueError(f"再実行LPの解が見つかりませんでした: {result.message}")

    return x

def generate_final_ranking_vector(X):
    row_sums = np.sum(X, axis=1)
    sorted_values = sorted(row_sums, reverse=True)
    rank_dict = {value: rank for rank, value in enumerate(sorted_values, start=1)}
    ranks = [rank_dict[x] for x in row_sums]
    return row_sums, np.array(ranks)

def calculate_kendall_tau_distance(R_0, R_mvr):
    N = len(R_mvr)
    D = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (R_mvr[i] < R_mvr[j] and R_0[i] > R_0[j]) or (R_mvr[i] > R_mvr[j] and R_0[i] < R_0[j]):
                D += 1
    return D

# メイン
if __name__ == "__main__":
    N = 50 # 候補者数
    M = 500  # 投票者数
    L_0 = 30  # ランキングの平均長さ
    # b = 1.0  # ノイズパラメータ
    b_values = [0.9, 0.91, 0.92, 0.93,0.94,0.95]  # ノイズパラメータの候補
    D_values = []

    for b in b_values:
        # データ生成
        phi, R_0, phi_prime = generate_synthetic_data(N, M, L_0, b)

        # 競争行列Aの計算
        A = compute_score_difference_matrix(phi_prime,N)

        final_solution = iterative_constraint_relaxation(A)
        # print("最終的なランキング行列X:")
        # print(final_solution)

        g, R_mvr = generate_final_ranking_vector(final_solution)
        # print("最終的なランキングベクトルR_mvr:")
        # print(R_mvr)

        # ケンドールタウ距離を計算
        D = calculate_kendall_tau_distance(R_mvr, R_0)
        D_values.append(D)
        print("ケンドールタウ距離D:")
        print(D)