import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import japanize_matplotlib 

# --- パラメータ設定 ---
# これらの値を変更して、様々な条件をシミュレーションできます。

M = 80.0      # クライマーの質量 (kg)
K = 1380.0    # ロープのばね定数 (N/m) - 硬いロープほど大きな値
D = 200.0     # ロープの減衰係数 (N·s/m) - 振動の収まりやすさ
G = 9.81      # 重力加速度 (m/s^2)
H = 1.0  # 墜落距離 (m)
ANCHOR_FACTOR = 1.7 # 支点にかかる荷重の係数（プーリー効果）

# --- 初期条件 ---
x0 = 0.0  # 初期位置 (m) - ロープが伸び始める瞬間
v0 = np.sqrt(2*G*H)     # 初期速度 (m/s) - 自由落下後のクライマーの速度
print(f"v0 = {v0}")

# --- シミュレーション時間 ---
t_span = (0, 6.0)  # 0秒から2秒後までを計算
t_eval = np.linspace(t_span[0], t_span[1], 500) # 計算する時間点

# --- 運動方程式の定義 ---
# m*x'' + d*x' + k*x = m*g  を変形して、連立微分方程式にする
# y[0] = x (位置), y[1] = v (速度)
def fall_dynamics(t, y, m, d, k, g):
    """
    クライマーの墜落に関する運動方程式を定義します。
    
    Args:
        t: 時刻
        y: 状態ベクトル [位置, 速度]
        m: 質量
        d: 減衰係数
        k: ばね定数
        g: 重力加速度
        
    Returns:
        状態ベクトルの時間微分 [速度, 加速度]
    """
    x, v = y
    dxdt = v
    dvdt = (m * g - d * v - k * x) / m
    return [dxdt, dvdt]

# --- 微分方程式を解く ---
# solve_ivp を使って数値的に解を求める
solution = solve_ivp(
    fall_dynamics,
    t_span,
    [x0, v0],
    args=(M, D, K, G),
    dense_output=True,
    t_eval=t_eval
)

# --- 結果の取得 ---
t = solution.t
x = solution.y[0]
v = solution.y[1]

# --- 荷重の計算 ---
# 荷重(張力) = ばねによる力 + 減衰による力
# T = k*x + d*v
climber_load = K * x + D * v
# 荷重が負になる場合は0とする（ロープは押せないので）
climber_load[climber_load < 0] = 0

# 支点にかかる荷重の計算
anchor_load = climber_load * ANCHOR_FACTOR

# --- グラフの描画 ---
fig, ax = plt.subplots(figsize=(10, 6))

# 荷重をkN単位でプロット
ax.plot(t, anchor_load / 1000, label='衝撃荷重', color='crimson', linewidth=2)

# グラフの装飾
ax.set_ylim(0, 3.5)
ax.set_title('クライミング墜落時の衝撃荷重シミュレーション', fontsize=16)
ax.set_xlabel('時間 (秒)', fontsize=12)
ax.set_ylabel('荷重 (kN)', fontsize=12)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# 最大荷重の点を表示
max_load_idx = np.argmax(anchor_load)
max_load_t = t[max_load_idx]
max_load_val = anchor_load[max_load_idx] / 1000
ax.plot(max_load_t, max_load_val, 'o', color='gold', markersize=8, markeredgecolor='black', label=f'最大荷重: {max_load_val:.2f} kN')

# 静止荷重（体重）の線を表示
static_load = (M * G) * ANCHOR_FACTOR / 1000
ax.axhline(y=static_load, color='gray', linestyle=':', label=f'停止時荷重: {static_load:.2f} kN')

# 凡例をまとめて表示
ax.legend()

plt.tight_layout()
plt.show()
