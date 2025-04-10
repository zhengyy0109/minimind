import os
import numpy as np
import tifffile as tiff
# import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ================== 物理参数 ==================
VOXEL_SIZE = 3.0e-6
TAU = 10.0
U_INLET = 1.0e-4
CS_SQ = 1.0 / 3.0
NU = CS_SQ * (TAU - 0.5)
G_INTER = -1e-4
G_WALL_OIL = 0.03
G_WALL_WATER = -0.03

print(f"=== 物理参数验证 ===")
print(f"黏度 NU: {NU:.3e} m²/s")
print(f"入口马赫数: {U_INLET / np.sqrt(CS_SQ):.4f}")

# ================== 模拟参数 ==================
NUM_STEPS = 2000
SAVE_INTERVAL = 100
OUTPUT_DIR = "output_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ramp_steps = 500

# ================== CT数据加载 ==================
tiff_dir = r"./120 tisu"
stack = []
for filename in sorted(os.listdir(tiff_dir)):
    if filename.startswith("xy2.view") and filename.endswith(".tif"):
        img = tiff.imread(os.path.join(tiff_dir, filename))
        stack.append(np.squeeze(img))
volume = np.stack(stack, axis=0)
Nz, Ny, Nx = volume.shape

# 转换到GPU
is_solid = np.asarray(volume == 0)
porosity = 1 - np.mean(is_solid)  #.get()
print(f"\n=== 孔隙结构 ===")
print(f"网格尺寸: {Nx}x{Ny}x{Nz}")
print(f"孔隙率: {porosity:.2%}")

# ================== D3Q19 速度集 ==================
c_cpu = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, -1, -1], [0, 1, -1]
], dtype=np.int8)

c = np.asarray(c_cpu)
w = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12, dtype=np.float64)

# 反弹方向索引
opp_dirs = [0, 2, 1, 4, 3, 6, 5, 9, 8, 7, 10, 13, 12, 11, 14, 18, 17, 16, 15]

# ================== MRT 矩阵 ==================
# M_d3q19 = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [-30, -11, -11, -11, -11, -11, -11, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
#     [12, -4, -4, -4, -4, -4, -4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0],
#     [0, -4, 4, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0],
#     [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, -1, 1],
#     [0, 0, 0, -4, 4, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, -1, 1],
#     [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1],
#     [0, 0, 0, 0, 0, -4, 4, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1],
#     [0, 2, 2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#     [0, -4, -4, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#     [0, 1, 1, 1, 1, -2, -2, 1, 1, 1, 1, -2, -2, -2, -2, 0, 0, 0, 0],
#     [0, -2, -2, -2, -2, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2, 0, 0, 0, 0],
#     [0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1],
#     [0, -2, -2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1],
#     [0, 0, 0, 1, 1, -2, -2, -1, -1, -1, -1, 2, 2, 2, 2, 0, 0, 0, 0],
#     [0, 0, 0, -2, -2, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, -1, 1, 1, 2, 2, -2, -2],
#     [0, 0, 0, 0, 0, 0, 0, -2, -2, 1, 1, -1, -1, 1, 1, 2, 2, -2, -2]
# ], dtype=np.float64)

M_d3q19 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [-30, -11, -11, -11, -11, -11, -11, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [12, -4, -4, -4, -4, -4, -4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, -1, 0, 0, 0, 0, 1, -1, -1, -1, 0, 0, 1, -1, 1, -1, 0, 0],
    [0, -4, 4, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0],
    [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1],
    [0, 0, 0, -4, 4, 0, 0, 1, 1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1],
    [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1],
    [0, 0, 0, 0, 0, -4, 4, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1],
    [0, 2, 2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, -4, -4, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 0, 0],
    [0, 0, 0, -2, -2, 2, 2, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, -1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, -1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1, 1, 1, -1]
], dtype=np.float64)

M_inv_d3q19 = np.linalg.inv(M_d3q19)
S = np.array([0, 1.2, 1.2, 0, 0, 0, 0, 0, 0, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2], dtype=np.float64)


# ================== 初始化 ==================
def initialize():
    rhoR = np.where(~is_solid, 0.85, 0.0)
    rhoB = np.where(~is_solid, 0.15, 0.0)
    u = np.zeros((Nz, Ny, Nx, 3), dtype=np.float64)

    fR = np.zeros((Nz, Ny, Nx, 19), dtype=np.float64)
    fB = np.zeros((Nz, Ny, Nx, 19), dtype=np.float64)
    for i in range(19):
        feqR = w[i] * rhoR
        feqB = w[i] * rhoB
        fR[..., i] = np.where(~is_solid, feqR, 0.0)
        fB[..., i] = np.where(~is_solid, feqB, 0.0)
    return fR, fB, u


fR, fB, u = initialize()


# ================== 核心函数 ==================
def equilibrium(rho, u):
    feq = np.zeros_like(fR)
    u_sq = (u ** 2).sum(axis=-1, keepdims=True)
    for i in range(19):
        cu = 3.0 * (c[i, 0] * u[..., 0] + c[i, 1] * u[..., 1] + c[i, 2] * u[..., 2])
        feq[..., i] = w[i] * rho * (1 + cu + 0.5 * cu ** 2 - 1.5 * u_sq.squeeze())
    return feq


def compute_force(rhoR, rhoB, is_solid):
    rhoR = np.where(~is_solid, rhoR, 0.0)
    rhoB = np.where(~is_solid, rhoB, 0.0)
    psiR = rhoR * (1 - np.exp(-rhoR))
    psiB = rhoB * (1 - np.exp(-rhoB))

    F_inter = np.zeros((Nz, Ny, Nx, 3), dtype=np.float64)
    for i in range(1, 19):
        dx, dy, dz = c_cpu[i]
        shifted_psiR = np.roll(psiR, (dz, dy, dx), (0, 1, 2))
        shifted_psiB = np.roll(psiB, (dz, dy, dx), (0, 1, 2))
        force_term = G_INTER * w[i] * c[i] * (shifted_psiR * shifted_psiB)[..., np.newaxis]
        F_inter += force_term.astype(np.float64)

    F_wall = (G_WALL_OIL * psiR[..., np.newaxis] + G_WALL_WATER * psiB[..., np.newaxis]) * is_solid[..., np.newaxis]
    return np.nan_to_num(F_inter + F_wall)


def mrt_collision(f, feq, force):
    shp = f.shape
    f_flat = f.reshape(-1, 19)
    feq_flat = feq.reshape(-1, 19)
    force_flat = force.reshape(-1, 3)

    m = f_flat @ M_d3q19.T
    m_eq = feq_flat @ M_d3q19.T

    F_m = np.zeros_like(m)
    F_m[:, 3] = (1 - 0.5 * S[3]) * force_flat[:, 0]
    F_m[:, 5] = (1 - 0.5 * S[5]) * force_flat[:, 1]
    F_m[:, 7] = (1 - 0.5 * S[7]) * force_flat[:, 2]

    m_post = m - S * (m - m_eq) + F_m
    return (m_post @ M_inv_d3q19.T).reshape(shp).clip(1e-6, 1e3)


def zou_he_boundary(f_phase, step):
    velocity_scale = min(1.0, step / ramp_steps)
    u_in = np.zeros((1, Ny, Nx, 3), dtype=np.float64)
    u_in[..., 2] = U_INLET * velocity_scale

    f_new = f_phase.copy()
    f_in = f_new[0]
    rho_in = np.sum(f_in, axis=-1).clip(1e-6, None)

    # 维度修正：添加通道维度
    feq_in = equilibrium(rho_in[..., None], u_in[0])  # 输入形状(Ny, Nx, 1)和(Ny, Nx, 3)

    for i in [5, 11, 12, 15, 16, 17, 18]:
        f_new[0, ..., i] = feq_in[..., i]
    return f_new


# ================== 主循环 ==================
for step in range(NUM_STEPS):
    try:
        # 1) 流步骤
        for i in range(19):
            dz, dy, dx = c_cpu[i]
            fR[..., i] = np.roll(fR[..., i], (dz, dy, dx), (0, 1, 2))
            fB[..., i] = np.roll(fB[..., i], (dz, dy, dx), (0, 1, 2))

        # 2) 固体反弹
        for i in range(19):
            opp = opp_dirs[i]
            fR[is_solid, i] = fR[is_solid, opp]
            fB[is_solid, i] = fB[is_solid, opp]

        # 3) 入口边界
        fR = zou_he_boundary(fR, step)
        fB = zou_he_boundary(fB, step)

        # 4) 计算宏观量
        rhoR = np.sum(fR, axis=-1).clip(1e-6, 5.0)
        rhoB = np.sum(fB, axis=-1).clip(1e-6, 5.0)
        density = (rhoR + rhoB).clip(1e-6, None)

        # 5) 计算力
        total_force = compute_force(rhoR, rhoB, is_solid)

        # 6) 计算速度（关键修复：正确einsum表达式）
        momentum = np.einsum('...i,ij->...j', fR + fB, c)  # 输入形状(Nz,Ny,Nx,19) × (19,3) → (Nz,Ny,Nx,3)
        u = (momentum + 0.5 * total_force) / density[..., None]
        u = np.clip(u, -1e-4, 1e-4)

        # 7) 碰撞步骤
        feqR = equilibrium(rhoR, u)
        feqB = equilibrium(rhoB, u)
        fR = mrt_collision(fR, feqR, total_force)
        fB = mrt_collision(fB, feqB, total_force)

        # 监控
        if step % 50 == 0:
            max_vel = np.nanmax(np.linalg.norm(u, axis=-1)).get()
            avg_rho = np.nanmean(density).get()
            print(f"Step {step:04d} | max|u| = {max_vel:.2e} | avg_rho = {avg_rho:.3f}")

        # 保存结果
        if step % SAVE_INTERVAL == 0:
            plt.figure(figsize=(10, 6))
            plt.imshow(rhoR.get()[Nz // 2], cmap='Reds', alpha=0.7)
            plt.imshow(rhoB.get()[Nz // 2], cmap='Blues', alpha=0.5)
            plt.contour(is_solid.get()[Nz // 2], colors='k', levels=[0.5], linewidths=0.5)
            plt.title(f"Step {step}")
            plt.colorbar(label="Density")
            plt.savefig(os.path.join(OUTPUT_DIR, f"step_{step:04d}.png"), dpi=150)
            plt.close()

    except Exception as e:
        print(f"\n!!! 模拟终止于第 {step} 步 !!!")
        print(f"错误原因: {str(e)}")
        break

print("\n模拟成功完成！请检查输出目录中的结果。")