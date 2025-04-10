import warnings
import numpy as np
import cupy as cp
from skimage import io
import pyvista as pv
import os
import gc
from cryptography.utils import CryptographyDeprecationWarning

# 忽略警告
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

# 设置 CuPy 内存池
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# ========== 物理参数 ==========
VOXEL_SIZE = 3.0e-6  # 体素大小 (m)
TAU = 10.0  # 松弛时间
U_INLET = 1.0e-4  # 入口速度 (m/s)
CS_SQ = 1.0 / 3.0  # 声速平方
NU = CS_SQ * (TAU - 0.5)  # 运动粘度
G_INTER = -1e-4  # 界面相互作用参数
G_WALL_OIL = 0.03  # 油相壁面相互作用参数
G_WALL_WATER = -0.03  # 水相壁面相互作用参数

# ========== D3Q19 模型参数 ==========
Q = 19  # 离散速度方向数

# 速度向量
c = cp.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1], [1, 1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, -1, 0], [1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1],
    [1, 0, -1], [-1, 0, -1], [0, 1, -1], [0, -1, -1]
], dtype=cp.float32)

# 权重系数
w = cp.array([
    1 / 3, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
    1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
    1 / 36, 1 / 36, 1 / 36, 1 / 36
], dtype=cp.float32)

# 声速
cs = cp.float32(1 / np.sqrt(3))

# ========== MRT 模型参数 ==========
# 转换矩阵
M = cp.array([
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
], dtype=cp.float32)

# 松弛矩阵
S = cp.diag(cp.array([
    1.19, 1.4, 1.4, 1.4, 1.4, 1.2, 1.2, 1.2, 1.2,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4
], dtype=cp.float32))

# 界面张力系数
sigma = cp.float32(0.1)


def read_tif_images(folder_path):
    """读取TIF格式的切片图像"""
    try:
        images = []
        for i in range(120):
            file_path = os.path.join(folder_path, f'xy2.view{i:03d}.tif')
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到文件: {file_path}")
            image = io.imread(file_path)
            image = image.astype(np.float32) / 255.0  # 归一化到0-1范围
            images.append(cp.array(image))
        return cp.stack(images, axis=2)  # 形状为 (x, y, z)
    except Exception as e:
        print(f"读取图像时出错: {str(e)}")
        raise


def equilibrium(rho, u):
    """计算平衡态分布函数"""
    try:
        # 确保维度正确
        rho = rho[..., cp.newaxis]  # 添加一个维度用于广播
        u = u.reshape(*u.shape[:-1], 1, 3)  # 确保速度场维度正确

        # 计算c·u
        cu = cp.einsum('...d,qd->...q', u, c)

        # 计算速度平方
        u_sq = cp.sum(u ** 2, axis=-1, keepdims=True)

        # 计算各项
        term1 = cu / cs ** 2
        term2 = (cu ** 2) / (2 * cs ** 4)
        term3 = u_sq / (2 * cs ** 2)

        # 计算平衡态分布函数
        f_eq = rho * w * (1 + term1 + term2 - term3)
        return f_eq
    except Exception as e:
        print(f"计算平衡态分布函数时出错: {str(e)}")
        raise


def compute_macroscopic(f):
    """计算宏观量（密度和速度）"""
    try:
        # 计算密度
        rho = cp.sum(f, axis=-1, keepdims=True)
        rho = cp.maximum(rho, 1e-10)  # 防止除以零

        # 计算速度
        u = cp.einsum('...q,qd->...d', f, c) / rho

        return rho[..., 0], u
    except Exception as e:
        print(f"计算宏观量时出错: {str(e)}")
        raise


def collision(f, rho, u):
    """碰撞步骤"""
    try:
        # 确保维度正确
        f = f.reshape(*f.shape[:-1], Q)
        rho = rho.reshape(*rho.shape[:-1], 1)
        u = u.reshape(*u.shape[:-1], 3)

        # 转换到矩空间
        m = cp.tensordot(M, f, axes=([1], [-1]))
        m = cp.moveaxis(m, 0, -1)

        # 计算平衡态矩
        f_eq = equilibrium(rho, u)
        m_eq = cp.tensordot(M, f_eq, axes=([1], [-1]))
        m_eq = cp.moveaxis(m_eq, 0, -1)

        # 松弛过程
        m_post = m - cp.einsum('q,...q->...q', cp.diag(S), (m - m_eq))

        # 转换回速度空间
        f_post = cp.tensordot(cp.linalg.inv(M), m_post, axes=([1], [-1]))
        f_post = cp.moveaxis(f_post, 0, -1)

        return f_post
    except Exception as e:
        print(f"碰撞步骤计算时出错: {str(e)}")
        raise


def streaming(f):
    """迁移步骤"""
    try:
        f = f.reshape(*f.shape[:-1], Q)
        for i in range(Q):
            f[..., i] = cp.roll(f[..., i], shift=tuple(c[i].astype(int)), axis=(0, 1, 2))
        return f
    except Exception as e:
        print(f"迁移步骤计算时出错: {str(e)}")
        raise


def compute_gradient(phi):
    """计算梯度（使用稳定的差分方法）"""
    try:
        nx, ny, nz = phi.shape
        grad = cp.zeros((3, nx, ny, nz), dtype=cp.float32)

        # x方向梯度
        if nx > 2:
            grad[0, 1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / 2.0
            grad[0, 0, :, :] = phi[1, :, :] - phi[0, :, :]
            grad[0, -1, :, :] = phi[-1, :, :] - phi[-2, :, :]
        elif nx > 1:
            grad[0, 0, :, :] = phi[1, :, :] - phi[0, :, :]
            grad[0, -1, :, :] = phi[-1, :, :] - phi[-2, :, :]

        # y方向梯度
        if ny > 2:
            grad[1, :, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / 2.0
            grad[1, :, 0, :] = phi[:, 1, :] - phi[:, 0, :]
            grad[1, :, -1, :] = phi[:, -1, :] - phi[:, -2, :]
        elif ny > 1:
            grad[1, :, 0, :] = phi[:, 1, :] - phi[:, 0, :]
            grad[1, :, -1, :] = phi[:, -1, :] - phi[:, -2, :]

        # z方向梯度
        if nz > 2:
            grad[2, :, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / 2.0
            grad[2, :, :, 0] = phi[:, :, 1] - phi[:, :, 0]
            grad[2, :, :, -1] = phi[:, :, -1] - phi[:, :, -2]
        elif nz > 1:
            grad[2, :, :, 0] = phi[:, :, 1] - phi[:, :, 0]
            grad[2, :, :, -1] = phi[:, :, -1] - phi[:, :, -2]

        return grad
    except Exception as e:
        print(f"计算梯度时出错: {str(e)}")
        raise


def compute_curvature(grad, unit_grad):
    """计算曲率（使用稳定的差分方法）"""
    try:
        nx, ny, nz = grad.shape[1:]
        kappa = cp.zeros((nx, ny, nz), dtype=cp.float32)

        # x方向曲率
        if nx > 2:
            kappa[1:-1, :, :] -= (unit_grad[0, 2:, :, :] - unit_grad[0, :-2, :, :]) / 2.0
            kappa[0, :, :] -= unit_grad[0, 1, :, :] - unit_grad[0, 0, :, :]
            kappa[-1, :, :] -= unit_grad[0, -1, :, :] - unit_grad[0, -2, :, :]
        elif nx > 1:
            kappa[0, :, :] -= unit_grad[0, 1, :, :] - unit_grad[0, 0, :, :]
            kappa[-1, :, :] -= unit_grad[0, -1, :, :] - unit_grad[0, -2, :, :]

        # y方向曲率
        if ny > 2:
            kappa[:, 1:-1, :] -= (unit_grad[1, :, 2:, :] - unit_grad[1, :, :-2, :]) / 2.0
            kappa[:, 0, :] -= unit_grad[1, :, 1, :] - unit_grad[1, :, 0, :]
            kappa[:, -1, :] -= unit_grad[1, :, -1, :] - unit_grad[1, :, -2, :]
        elif ny > 1:
            kappa[:, 0, :] -= unit_grad[1, :, 1, :] - unit_grad[1, :, 0, :]
            kappa[:, -1, :] -= unit_grad[1, :, -1, :] - unit_grad[1, :, -2, :]

        # z方向曲率
        if nz > 2:
            kappa[:, :, 1:-1] -= (unit_grad[2, :, :, 2:] - unit_grad[2, :, :, :-2]) / 2.0
            kappa[:, :, 0] -= unit_grad[2, :, :, 1] - unit_grad[2, :, :, 0]
            kappa[:, :, -1] -= unit_grad[2, :, :, -1] - unit_grad[2, :, :, -2]
        elif nz > 1:
            kappa[:, :, 0] -= unit_grad[2, :, :, 1] - unit_grad[2, :, :, 0]
            kappa[:, :, -1] -= unit_grad[2, :, :, -1] - unit_grad[2, :, :, -2]

        return kappa
    except Exception as e:
        print(f"计算曲率时出错: {str(e)}")
        raise


def color_gradient_force(f1, f2):
    """计算界面力"""
    try:
        # 计算宏观量
        rho1, u1 = compute_macroscopic(f1)
        rho2, u2 = compute_macroscopic(f2)

        # 计算序参数
        rho = rho1 + rho2 + 1e-10
        phi = (rho1 - rho2) / rho

        # 计算梯度
        grad_phi = compute_gradient(phi.squeeze(3))

        # 计算梯度模
        grad_phi_sq = cp.sum(grad_phi ** 2, axis=0)
        grad_phi_norm = cp.sqrt(grad_phi_sq + 1e-10)
        unit_grad = grad_phi / (grad_phi_norm + 1e-10)

        # 计算曲率
        kappa = compute_curvature(grad_phi, unit_grad)

        # 计算力
        force = -sigma * kappa * grad_phi / (grad_phi_norm + 1e-10)
        return force[..., cp.newaxis]  # 增加一个维度与速度对齐
    except Exception as e:
        print(f"计算界面力时出错: {str(e)}")
        raise


def boundary_conditions(f1, f2):
    """边界条件处理"""
    try:
        # 确保维度正确
        f1 = f1.reshape(*f1.shape[:-1], Q)
        f2 = f2.reshape(*f2.shape[:-1], Q)

        # 周期性边界条件
        for axis in (0, 1, 2):
            f1 = cp.roll(f1, shift=1, axis=axis)
            f2 = cp.roll(f2, shift=1, axis=axis)
        return f1, f2
    except Exception as e:
        print(f"边界条件处理时出错: {str(e)}")
        raise


def wetting_treatment(f1, f2, theta_deg):
    """润湿性处理"""
    try:
        # 计算宏观量
        rho1, u1 = compute_macroscopic(f1)
        rho2, u2 = compute_macroscopic(f2)

        # 计算序参数
        rho = rho1 + rho2 + 1e-10
        phi = (rho1 - rho2) / rho

        # 计算梯度
        grad_phi = compute_gradient(phi.squeeze(3))
        grad_phi_norm = cp.linalg.norm(grad_phi, axis=0) + 1e-10
        unit_grad = grad_phi / grad_phi_norm

        # 固体表面法向量
        normal = cp.zeros_like(grad_phi)
        normal[2] = 1.0

        # 接触角调整
        theta = np.deg2rad(theta_deg)
        new_grad = (cp.cos(theta) * normal +
                    cp.sin(theta) * (unit_grad - cp.sum(unit_grad * normal, axis=0) * normal))
        new_grad_norm = cp.linalg.norm(new_grad, axis=0) + 1e-10
        new_unit_grad = new_grad / new_grad_norm

        # 重构梯度
        adjusted_grad = new_unit_grad * grad_phi_norm
        c_dot_grad = cp.einsum('qd,...d->...q', c, adjusted_grad)

        # 更新分布函数
        f1 = f1.reshape(*f1.shape[:-1], Q)
        f2 = f2.reshape(*f2.shape[:-1], Q)
        f1 += 0.1 * w * c_dot_grad
        f2 += 0.1 * w * c_dot_grad

        return f1, f2
    except Exception as e:
        print(f"润湿性处理时出错: {str(e)}")
        raise


def save_vtk(f1, f2, step, output_folder):
    """保存VTK格式的结果文件"""
    try:
        rho1, _ = compute_macroscopic(f1)
        rho2, _ = compute_macroscopic(f2)

        grid = pv.ImageData()
        grid.dimensions = np.array(rho1.shape) + 1
        grid.spacing = [VOXEL_SIZE] * 3

        grid.point_data['rho1'] = rho1.get().flatten(order='F')
        grid.point_data['rho2'] = rho2.get().flatten(order='F')

        output_path = os.path.join(output_folder, f"step_{step:04d}.vtk")
        grid.save(output_path)
        print(f"保存 VTK 文件: {output_path}")
    except Exception as e:
        print(f"保存VTK文件时出错: {str(e)}")
        raise


def main(input_folder, output_folder, num_steps=500, save_interval=50):
    """主模拟循环"""
    try:
        # 检查输入输出文件夹
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 读取初始场
        print("正在读取初始场...")
        rho1_initial = read_tif_images(input_folder)
        rho2_initial = 1.0 - rho1_initial

        # 确保初始场的维度正确
        rho1_initial = rho1_initial.reshape(*rho1_initial.shape, 1)
        rho2_initial = rho2_initial.reshape(*rho2_initial.shape, 1)

        # 初始化速度场
        u_initial = cp.zeros((*rho1_initial.shape[:-1], 3), dtype=cp.float32)

        # 初始化分布函数
        print("正在初始化分布函数...")
        f1 = equilibrium(rho1_initial, u_initial)
        f2 = equilibrium(rho2_initial, u_initial)

        # 释放不需要的内存
        del rho1_initial, rho2_initial, u_initial
        gc.collect()
        cp.cuda.Stream.null.synchronize()

        for step in range(num_steps):
            try:
                # 计算宏观量
                rho1, u1 = compute_macroscopic(f1.squeeze(3))
                rho2, u2 = compute_macroscopic(f2.squeeze(3))

                # 计算界面力
                force = color_gradient_force(f1, f2)

                # 添加外力项
                u1 += force * TAU
                u2 += force * TAU

                # 碰撞步骤
                f1 = collision(f1, rho1, u1)
                f2 = collision(f2, rho2, u2)

                # 迁移步骤
                f1 = streaming(f1)
                f2 = streaming(f2)

                # 润湿处理
                f1, f2 = wetting_treatment(f1, f2, theta_deg=30)

                # 边界条件
                f1, f2 = boundary_conditions(f1, f2)

                # 保存结果
                if step % save_interval == 0:
                    print(f"Step {step}/{num_steps}")
                    save_vtk(f1, f2, step, output_folder)

                # 定期清理内存
                if step % 10 == 0:
                    gc.collect()
                    cp.cuda.Stream.null.synchronize()

            except Exception as e:
                print(f"步骤 {step} 执行出错: {str(e)}")
                continue

    except Exception as e:
        print(f"模拟过程出错: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        input_folder = "./120 tisu"
        output_folder = "simulation_results"

        if not os.path.exists(input_folder):
            print(f"错误: 输入文件夹 {input_folder} 不存在")
            print("请确保输入文件夹存在并包含所需的TIF文件")
            exit(1)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"创建输出文件夹: {output_folder}")

        main(input_folder, output_folder, num_steps=500, save_interval=50)
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        exit(1)