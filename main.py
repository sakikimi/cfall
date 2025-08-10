import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import japanize_matplotlib
import sys
import pandas as pd
import os

def run_simulation():
    # --- パラメータ設定 ---
    M = 80.0      # クライマーの質量 (kg)
    K = 1380.0    # ロープのばね定数 (N/m)
    D = 200.0     # ロープの減衰係数 (N·s/m)
    G = 9.81      # 重力加速度 (m/s^2)
    H = 1.0       # 墜落距離 (m)
    ANCHOR_FACTOR = 1.7 # 支点にかかる荷重の係数（プーリー効果）

    # --- 初期条件 ---
    x0 = 0.0
    v0 = np.sqrt(2 * G * H)
    print(f"初速度 v0 = {v0:.2f} m/s")

    # --- シミュレーション時間 ---
    t_span = (0, 4.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # --- 運動方程式の定義 ---
    def fall_dynamics(t, y, m, d, k, g):
        x, v = y
        dxdt = v
        dvdt = (m * g - d * v - k * x) / m
        return [dxdt, dvdt]

    # --- 微分方程式を解く ---
    solution = solve_ivp(
        fall_dynamics,
        t_span,
        [x0, v0],
        args=(M, D, K, G),
        dense_output=True,
        t_eval=t_eval
    )

    # --- 結果の取得と荷重計算 ---
    t_sim = solution.t
    x_sim = solution.y[0]
    v_sim = solution.y[1]
    
    climber_load_sim = K * x_sim + D * v_sim
    climber_load_sim[climber_load_sim < 0] = 0
    anchor_load_sim = climber_load_sim * ANCHOR_FACTOR
    
    # 静止荷重の計算
    static_load = (M * G) * ANCHOR_FACTOR
    
    return t_sim, anchor_load_sim, static_load

def plot_data(t_sim, anchor_load_sim, static_load, csv_filepath=None):
    """シミュレーション結果と実測値をプロットする関数"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # シミュレーション結果をプロット
    ax.plot(t_sim, anchor_load_sim / 1000, label='シミュレーション荷重', color='crimson', linewidth=2)

    # CSVファイルが指定されていれば、実測値もプロット
    if csv_filepath:
        try:
            # CSVを読み込む（1行目はヘッダーとして扱う）
            data = pd.read_csv(csv_filepath)
            t_meas = data.iloc[:, 0].values
            load_meas = data.iloc[:, 1].values

            # ピーク位置で時間軸を合わせる
            peak_idx_sim = np.argmax(anchor_load_sim)
            t_peak_sim = t_sim[peak_idx_sim]

            peak_idx_meas = np.argmax(load_meas)
            t_peak_meas = t_meas[peak_idx_meas]

            time_offset = t_peak_sim - t_peak_meas
            t_meas_aligned = t_meas + time_offset
            
            # 実測値をプロット
            ax.plot(t_meas_aligned, load_meas, label='実測荷重', linestyle='none', marker='o', markerfacecolor='white', markeredgecolor='k', markersize=6)

            title = f'シミュレーションと実測値の比較 ({os.path.basename(csv_filepath)})'
        except Exception as e:
            print(f"エラー: CSVファイルの読み込みまたは処理に失敗しました。 {e}")
            title = 'クライミング墜落時の衝撃荷重シミュレーション'
    else:
        title = 'クライミング墜落時の衝撃荷重シミュレーション'

    # --- グラフの装飾 ---
    ax.set_ylim(0, max(np.max(anchor_load_sim) / 1000 * 1.1, 3.5))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('時間 (秒)', fontsize=12)
    ax.set_ylabel('荷重 (kN)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 最大荷重の点を表示 (シミュレーション基準)
    # max_load_idx = np.argmax(anchor_load_sim)
    # max_load_t = t_sim[max_load_idx]
    # max_load_val = anchor_load_sim[max_load_idx] / 1000
    # ax.plot(max_load_t, max_load_val, 'o', color='gold', markersize=8, markeredgecolor='black', label=f'シミュレーション最大荷重: {max_load_val:.2f} kN')

    # 静止荷重の線を表示
    ax.axhline(y=static_load / 1000, color='gray', linestyle=':', label=f'停止時荷重: {static_load/1000:.2f} kN')

    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # シミュレーションを実行
    t_sim, anchor_load_sim, static_load = run_simulation()

    # コマンドライン引数をチェック
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None

    # グラフを描画
    plot_data(t_sim, anchor_load_sim, static_load, csv_filepath=csv_file)
