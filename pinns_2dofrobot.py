# -*- coding: utf-8 -*-
"""
PINNS_optimized.ipynb

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from google.colab import files

torch.manual_seed(42)
np.random.seed(42)

model_path = 'pinn_robot_model.pt'


# =====================================================================
# 1. GROUND TRUTH ROBOT SIMULATOR
# =====================================================================
class GroundTruthRobot:
    def __init__(self):
        self.m1, self.m2 = 1.0, 0.8
        self.l1, self.l2 = 0.5, 0.5
        self.g = 9.81
        self.d1, self.d2 = 0.3, 0.2

    def mass_matrix(self, q):
        q2 = q[1]
        m11 = self.m1 * self.l1**2 + self.m2 * (self.l1**2 + self.l2**2 + 2 * self.l1 * self.l2 * torch.cos(q2))
        m12 = self.m2 * (self.l2**2 + self.l1 * self.l2 * torch.cos(q2))
        m22 = torch.tensor(self.m2 * self.l2**2, device=q.device, dtype=q.dtype)
        return torch.stack([m11, m12, m12, m22]).reshape(2, 2)

    def potential_energy(self, q):
        q1, q2 = q[0], q[1]
        return (self.m1 + self.m2) * self.g * self.l1 * torch.cos(q1) + self.m2 * self.g * self.l2 * torch.cos(q1 + q2)

    def damping_matrix(self, q):
        return torch.diag(torch.tensor([self.d1, self.d2], device=q.device, dtype=q.dtype))

    def actuation_matrix(self, q):
        q2 = q[1]
        a11 = 1.0 + 0.1 * torch.sin(q2)
        a12 = torch.tensor(0.05, device=q.device, dtype=q.dtype)
        a21 = torch.tensor(0.02, device=q.device, dtype=q.dtype)
        a22 = 0.8 + 0.1 * torch.cos(q2)
        return torch.stack([a11, a12, a21, a22]).reshape(2, 2)

    def coriolis_force(self, q, dq):
        h = -self.m2 * self.l1 * self.l2 * torch.sin(q[1])
        C = torch.stack([h * dq[1], h * (dq[0] + dq[1]), -h * dq[0], torch.tensor(0.0, device=q.device, dtype=q.dtype)]).reshape(2, 2)
        return torch.matmul(C, dq)

    def forward_dynamics(self, q, dq, u):
        q_req = q.detach().clone().requires_grad_(True)
        V = self.potential_energy(q_req)
        G = torch.autograd.grad(V, q_req, create_graph=False)[0]

        M = self.mass_matrix(q)
        D = self.damping_matrix(q)
        A = self.actuation_matrix(q)
        C_dq = self.coriolis_force(q, dq)

        forces = torch.matmul(A, u.detach()) - C_dq - G - torch.matmul(D, dq)
        ddq = torch.linalg.solve(M, forces)
        return ddq.detach()

# =====================================================================
# 2. FULLY VECTORIZED PINN SUB-NETWORKS
# =====================================================================

class VectorMassNetwork(nn.Module):
    def __init__(self, dim=2, hidden_dim=32, eps=1e-3):
        super().__init__()
        self.dim, self.eps = dim, eps
        out_dim = (dim * (dim + 1)) // 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, q):
        out = self.net(q)
        B = q.shape[0]
        L = torch.zeros(B, self.dim, self.dim, device=q.device, dtype=q.dtype)
        L[:, 0, 0] = torch.nn.functional.softplus(out[:, 0]) + self.eps
        L[:, 1, 1] = torch.nn.functional.softplus(out[:, 1]) + self.eps
        L[:, 1, 0] = out[:, 2]
        return torch.bmm(L, L.transpose(1, 2))

class VectorPotentialNetwork(nn.Module):
    def __init__(self, dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, q):
        return self.net(q).squeeze(-1)

class VectorDampingNetwork(nn.Module):
    def __init__(self, dim=2, hidden_dim=32):
        super().__init__()
        self.dim = dim
        out_dim = (dim * (dim + 1)) // 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, q):
        out = self.net(q)
        B = q.shape[0]
        L = torch.zeros(B, self.dim, self.dim, device=q.device, dtype=q.dtype)
        L[:, 0, 0] = torch.nn.functional.softplus(out[:, 0])
        L[:, 1, 1] = torch.nn.functional.softplus(out[:, 1])
        L[:, 1, 0] = out[:, 2]
        return torch.bmm(L, L.transpose(1, 2))

class VectorActuatorNetwork(nn.Module):
    def __init__(self, dim=2, hidden_dim=32):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, dim * dim)
        )
    def forward(self, q):
        out = self.net(q)
        return out.reshape(-1, self.dim, self.dim)

class BatchedPINN(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.m_nn = VectorMassNetwork(dim)
        self.p_nn = VectorPotentialNetwork(dim)
        self.d_nn = VectorDampingNetwork(dim)
        self.a_nn = VectorActuatorNetwork(dim)

    def compute_acceleration_batch(self, q, dq, u):
        q_req = q.clone().detach().requires_grad_(True)
        M = self.m_nn(q_req)
        V = self.p_nn(q_req)
        D = self.d_nn(q_req)
        A = self.a_nn(q_req)

        # Compute independent graph directly without memory leaks
        G = torch.autograd.grad(V.sum(), q_req, create_graph=True)[0]

        # Momentum vector calculations
        p = torch.bmm(M, dq.unsqueeze(-1)).squeeze(-1)

        # Intermediate retaining limits leaks to single backward pass
        p1_grad = torch.autograd.grad(p[:, 0].sum(), q_req, create_graph=True, retain_graph=True)[0]
        p2_grad = torch.autograd.grad(p[:, 1].sum(), q_req, create_graph=True, retain_graph=True)[0]

        dp1_dt = torch.sum(p1_grad * dq, dim=1, keepdim=True)
        dp2_dt = torch.sum(p2_grad * dq, dim=1, keepdim=True)
        dp_dq_dq = torch.cat([dp1_dt, dp2_dt], dim=1)

        T = 0.5 * torch.sum(dq * p, dim=1)

        # Final gradient natively frees the graph cleanly
        dT_dq = torch.autograd.grad(T.sum(), q_req, create_graph=True)[0]

        C_dq = dp_dq_dq - dT_dq
        Au = torch.bmm(A, u.unsqueeze(-1)).squeeze(-1)
        D_dq = torch.bmm(D, dq.unsqueeze(-1)).squeeze(-1)

        forces = Au - D_dq - G - C_dq
        ddq = torch.linalg.solve(M, forces.unsqueeze(-1)).squeeze(-1)
        return ddq

    def rk4_step_batch(self, q, dq, u, dt):
        f = lambda q_c, dq_c: self.compute_acceleration_batch(q_c, dq_c, u)

        k1_q = dq
        k1_v = f(q, dq)

        k2_q = dq + 0.5 * dt * k1_v
        k2_v = f(q + 0.5 * dt * k1_q, k2_q)

        k3_q = dq + 0.5 * dt * k2_v
        k3_v = f(q + 0.5 * dt * k2_q, k3_q)

        k4_q = dq + dt * k3_v
        k4_v = f(q + dt * k3_q, k4_q)

        q_next = q + (dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        dq_next = dq + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        return q_next, dq_next

# =====================================================================
# 3. DATASET GENERATION (Standardized to RK4)
# =====================================================================

def generate_dataset_tensor(robot, num_trajectories=20, steps_per_traj=80, dt=0.01):
    q_list, dq_list, u_list, q_next_list, dq_next_list = [], [], [], [], []
    for _ in range(num_trajectories):
        q = torch.randn(2) * 0.5
        dq = torch.randn(2) * 0.2
        for _ in range(steps_per_traj):
            u = torch.randn(2) * 2.0

            k1_q = dq
            k1_v = robot.forward_dynamics(q, dq, u)
            k2_q = dq + 0.5 * dt * k1_v
            k2_v = robot.forward_dynamics(q + 0.5 * dt * k1_q, k2_q, u)
            k3_q = dq + 0.5 * dt * k2_v
            k3_v = robot.forward_dynamics(q + 0.5 * dt * k2_q, k3_q, u)
            k4_q = dq + dt * k3_v
            k4_v = robot.forward_dynamics(q + dt * k3_q, k4_q, u)

            q_next = q + (dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
            dq_next = dq + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

            q_list.append(q.clone())
            dq_list.append(dq.clone())
            u_list.append(u.clone())
            q_next_list.append(q_next.clone())
            dq_next_list.append(dq_next.clone())
            q, dq = q_next, dq_next

    return (torch.stack(q_list), torch.stack(dq_list), torch.stack(u_list),
            torch.stack(q_next_list), torch.stack(dq_next_list))

# =====================================================================
# 4. ROBUST COMPUTED TORQUE CONTROLLER
# =====================================================================

def tracking_controller(pinn, q, dq, q_ref, dq_ref, ddq_ref, Kp, Kd):
    q_b = q.unsqueeze(0)
    dq_b = dq.unsqueeze(0)

    # 1. Compute zero-actuation acceleration
    # We must allow graph creation here so pinn.compute_acceleration_batch
    # can calculate the Coriolis and Gravity gradients internally.
    u_zero = torch.zeros((1, 2), device=q.device, dtype=q.dtype)
    ddq_zero = pinn.compute_acceleration_batch(q_b, dq_b, u_zero)[0].detach()

    # 2. Extract Mass and Actuation matrices
    with torch.no_grad():
        M = pinn.m_nn(q_b)[0]
        A = pinn.a_nn(q_b)[0]

    # n(q, dq) = C*dq + G + D*dq (Derived from M * ddq_zero = -n)
    n = -torch.matmul(M, ddq_zero)

    # Standard Computed Torque Control (CTC) Error
    e = q_ref - q
    de = dq_ref - dq
    v = ddq_ref + torch.matmul(Kp, e) + torch.matmul(Kd, de)

    # Control Law: u = A_pinv * (M * v + n)
    A_pinv = torch.linalg.pinv(A, rcond=1e-4)
    u = torch.matmul(A_pinv, torch.matmul(M, v) + n)

    return torch.clamp(u, -30.0, 30.0)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 5. DEMO & VISUALIZATION
# =====================================================================

if __name__ == "__main__":
    robot = GroundTruthRobot()
    pinn = BatchedPINN(dim=2)
    dt = 0.01

    if os.path.exists(model_path):
        os.remove(model_path)

    print("Generating training dataset...")
    q_all, dq_all, u_all, q_next_all, dq_next_all = generate_dataset_tensor(robot, num_trajectories=20, steps_per_traj=80, dt=dt)

    optimizer = optim.Adam(pinn.parameters(), lr=5e-3)
    batch_size = 512
    num_samples = q_all.shape[0]
    epochs = 10000

    print(f"Starting FAST Vectorized Training for {epochs} epochs on CPU...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        num_batches = 0
        for b in range(0, num_samples, batch_size):
            idx = perm[b : b + batch_size]
            q_b, dq_b, u_b = q_all[idx], dq_all[idx], u_all[idx]
            q_next_b, dq_next_b = q_next_all[idx], dq_next_all[idx]

            optimizer.zero_grad()
            q_pred, dq_pred = pinn.rk4_step_batch(q_b, dq_b, u_b, dt)
            loss = torch.mean((q_next_b - q_pred)**2) + torch.mean((dq_next_b - dq_pred)**2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {total_loss / num_batches:.8f} | Elapsed: {time.time() - t_start:.2f}s")

    print(f"Training finished in {time.time() - t_start:.2f} SECONDS!")
    torch.save(pinn.state_dict(), model_path)
    print(f"Model saved locally as '{model_path}'.")

    # Trigger browser download automatically
    print("Downloading 'pinn_robot_model.pt' to your computer...")
    files.download(model_path)


    # Simulation setup
    sim_time = 5.0
    steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, steps)

    q_actual = torch.tensor([0.2, -0.2], dtype=torch.float32)
    dq_actual = torch.tensor([0.0, 0.0], dtype=torch.float32)

    q_hist, q_ref_hist = [], []
    Kp = torch.diag(torch.tensor([120.0, 120.0], dtype=torch.float32))
    Kd = torch.diag(torch.tensor([20.0, 20.0], dtype=torch.float32))

    print("\nRunning Trajectory Tracking Control Demo...")
    for step in range(steps):
        t_curr = t[step]
        q_ref = torch.tensor([0.5 * np.sin(1.5 * t_curr), 0.4 * np.cos(1.5 * t_curr)], dtype=torch.float32)
        dq_ref = torch.tensor([0.75 * np.cos(1.5 * t_curr), -0.6 * np.sin(1.5 * t_curr)], dtype=torch.float32)
        ddq_ref = torch.tensor([-1.125 * np.sin(1.5 * t_curr), -0.9 * np.cos(1.5 * t_curr)], dtype=torch.float32)

        u = tracking_controller(pinn, q_actual, dq_actual, q_ref, dq_ref, ddq_ref, Kp, Kd)

        ddq = robot.forward_dynamics(q_actual, dq_actual, u)
        dq_actual = dq_actual + dt * ddq
        q_actual = q_actual + dt * dq_actual

        q_hist.append(q_actual.numpy().copy())
        q_ref_hist.append(q_ref.numpy().copy())

    q_hist = np.array(q_hist)
    q_ref_hist = np.array(q_ref_hist)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(t, q_ref_hist[:, 0], 'r--', linewidth=2, label=r'Reference $q_1$')
    axes[0].plot(t, q_hist[:, 0], 'b-', linewidth=1.5, label=r'PINN Control $q_1$')
    axes[0].set_ylabel('Joint 1 Angle [rad]', fontsize=11)
    axes[0].set_title('PINN-Based Non-Linear Robot Trajectory Control', fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='upper right')

    axes[1].plot(t, q_ref_hist[:, 1], 'r--', linewidth=2, label=r'Reference $q_2$')
    axes[1].plot(t, q_hist[:, 1], 'g-', linewidth=1.5, label=r'PINN Control $q_2$')
    axes[1].set_xlabel('Time [s]', fontsize=11)
    axes[1].set_ylabel('Joint 2 Angle [rad]', fontsize=11)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('pinn_robot_control_demo.png', dpi=300)
    print("Demo finished successfully! Saved plot to 'pinn_robot_control_demo.png'.")
    plt.show(block=True)
