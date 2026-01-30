import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import mujoco
from g1_arm_rl_env import G1ArmReachEnv
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# ============================================================
# Core Control Primitives
# ============================================================
from qpsolvers import solve_qp

def solve_joint_velocity_qp(J, v_des, dq_null, dq_prev, dt, dq_bounds):
    """
    QP: Minimize
        || J dq - v_des ||^2 + lambda * || dq - dq_null ||^2
    subject to:
        dq_min <= dq <= dq_max
        (optionally) other linear constraints
    """
    n = J.shape[1]
    lambda_null = 0.05  # nullspace regularization

    # Quadratic objective
    H = 2 * (J.T @ J + lambda_null * np.eye(n))
    f = -2 * (J.T @ v_des + lambda_null * dq_null)

    # Joint velocity limits (box constraints)
    dq_min, dq_max = dq_bounds

    # qpsolvers expects H, f, A, b, G, h, lb, ub
    # Here, no inequality constraints, just box constraints
    dq_sol = solve_qp(H, f, None, None, None, None, dq_min, dq_max, solver="osqp")

    # If QP fails (e.g. numerical issue), fall back to pseudo-inverse solution
    if dq_sol is None:
        dq_sol = dq_prev if dq_prev is not None else np.zeros(n)
    return dq_sol

def task_space_impedance(p, v, p_goal, Kp=80.0, Kd=15.0):
    return Kp * (p_goal - p) - Kd * v


def damped_jacobian_pinv(J, damping=1e-2):
    JJt = J @ J.T
    return J.T @ np.linalg.inv(JJt + damping * np.eye(JJt.shape[0]))


def nullspace_control(q, dq, q_rest, K_null=0.4, Kd_null=0.3):
    return -K_null * (q - q_rest) - Kd_null * dq


def jerk_regularization(dq, dq_prev, K_jerk=0.15):
    if dq_prev is None:
        return dq
    return dq - K_jerk * (dq - dq_prev)


def rate_limit(x, x_prev, max_rate, dt):
    if x_prev is None:
        return x
    dx = x - x_prev
    dx = np.clip(dx, -max_rate * dt, max_rate * dt)
    return x_prev + dx


def soft_clip(x, limit):
    return limit * np.tanh(x / limit)


# ============================================================
# Phase-aware authority & smoothing
# ============================================================

def task_authority_scale(dist, d_far=0.30, d_near=0.05):
    if dist > d_far:
        return 0.6
    if dist < d_near:
        return 1.2
    w = (d_far - dist) / (d_far - d_near)
    return 0.6 + 0.6 * w


def adaptive_lowpass(x, x_prev, dist,
                     alpha_far=0.95,
                     alpha_near=0.6,
                     d_near=0.05):
    if x_prev is None:
        return x
    alpha = alpha_near if dist < d_near else alpha_far
    return alpha * x_prev + (1.0 - alpha) * x


def torso_safety_scale(p, torso_pos=np.array([0.0, 0.0, 0.90]), safe_radius=0.25):
    d = np.linalg.norm(p - torso_pos)
    if d >= safe_radius:
        return 1.0
    return np.clip(d / safe_radius, 0.2, 1.0)


# ============================================================
# Analytic Jacobian (MuJoCo)
# ============================================================

# def compute_jacobian_analytic(env, q, body_name="right_wrist_yaw_link"):
#     for idx, val in zip(env._qadr, q):
#         env.data.qpos[idx] = val
#     mujoco.mj_forward(env.model, env.data)

#     jacp = np.zeros((3, env.model.nv))
#     jacr = np.zeros((3, env.model.nv))

#     body_id = mujoco.mj_name2id(
#         env.model,
#         mujoco.mjtObj.mjOBJ_BODY,
#         body_name
#     )
#     if body_id < 0:
#         raise ValueError(f"Body '{body_name}' not found")

#     mujoco.mj_jacBody(env.model, env.data, jacp, jacr, body_id)

#     return jacp[:, env._qadr]

def compute_jacobian(env, q, eps=1e-4):
    for idx, val in zip(env._qadr, q):
        env.data.qpos[idx] = val
    env._mujoco.mj_forward(env.model, env.data)

    p0 = env._fk()
    J = np.zeros((3, len(q)))

    for i in range(len(q)):
        dq = q.copy()
        dq[i] += eps
        for idx, val in zip(env._qadr, dq):
            env.data.qpos[idx] = val
        env._mujoco.mj_forward(env.model, env.data)
        pi = env._fk()
        J[:, i] = (pi - p0) / eps

    return J

def compute_jacobian_point(env, q, body_name, point_offset):
    """
    Analytic Jacobian of an arbitrary point rigidly attached to a body.
    point_offset is expressed in the BODY frame.
    """

    for idx, val in zip(env._qadr, q):
        env.data.qpos[idx] = val
    mujoco.mj_forward(env.model, env.data)

    jacp = np.zeros((3, env.model.nv))
    jacr = np.zeros((3, env.model.nv))

    body_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )

    mujoco.mj_jacBody(env.model, env.data, jacp, jacr, body_id)

    # World rotation of body
    R = env.data.xmat[body_id].reshape(3, 3)

    # Offset in world frame
    r = R @ np.asarray(point_offset)

    # Correct linear Jacobian: Jp_point = Jp_body + ω × r
    Jp = jacp + np.cross(jacr.T, r).T

    # Extract arm DOFs
    arm_dofs = [env.model.jnt_dofadr[j] for j in env._arm_joint_ids]
    return Jp[:, arm_dofs]


# ============================================================
# Experiment Setup
# ============================================================

N_EPISODES = 100
MAX_STEPS = 400

env = G1ArmReachEnv(render_mode="human")
env._arm_joint_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

env._arm_joint_ids = [
    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in env._arm_joint_names
]

# import mujoco

# print("\n=== MuJoCo MODEL INTROSPECTION ===")

# print(f"Number of sites: {env.model.nsite}")
# print("Site names:")
# for i in range(env.model.nsite):
#     name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, i)
#     print(f"  {i}: {name}")

# print(f"\nNumber of bodies: {env.model.nbody}")
# print("Body names:")
# for i in range(env.model.nbody):
#     name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i)
#     print(f"  {i}: {name}")

# print(f"\nNumber of joints: {env.model.njnt}")
# print("Joint names:")
# for i in range(env.model.njnt):
#     name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, i)
#     print(f"  {i}: {name}")

# print("=================================\n")


q_slice = slice(0, 7)
dq_slice = slice(7, 14)
p_hand_slice = slice(14, 17)
p_goal_slice = slice(17, 20)

all_final_dists = []
all_joint_delta_rates = []

# ============================================================
# Rollouts
# ============================================================
env.action_space.seed(42)
for ep in range(N_EPISODES):
    # obs, _ = env.reset()
    obs, _ = env.reset(seed=42 + ep)

    prev_dq_raw = None
    prev_dq_cmd = None
    rates = []
    for step in range(MAX_STEPS):
        q = obs[q_slice]
        dq = obs[dq_slice]
        p_hand = obs[p_hand_slice]
        p_goal = obs[p_goal_slice]
        dt = env.model.opt.timestep

        dist = np.linalg.norm(p_goal - p_hand)

        # --- Jacobian ---
        J = compute_jacobian_point(
            env,
            q,
            body_name="right_wrist_yaw_link",
            point_offset=[0.0415, -0.003, 0.0]
        )
        v_hand = J @ dq

        # --- Task-space impedance ---
        F = task_space_impedance(p_hand, v_hand, p_goal)
        F *= task_authority_scale(dist)

        # --- Task → joint via QP ---
        dq_null = nullspace_control(q, dq, q_rest=np.zeros_like(q))

        # Joint velocity bounds (use 80% of action space limits for safety)
        dq_lim = 0.8 * env.action_space.high
        dq_min = -np.abs(dq_lim)
        dq_max = +np.abs(dq_lim)
        dq_bounds = (dq_min, dq_max)

        # Solve QP for joint velocity command
        dq_cmd = solve_joint_velocity_qp(
            J,                # Jacobian (3 x 7)
            F,                # desired task velocity (3,)
            dq_null,          # nullspace reference (7,)
            prev_dq_cmd,      # for fallback
            dt,
            dq_bounds         # box constraints (min, max)
        )

        # --- Smoothness ---
        dq_cmd = jerk_regularization(dq_cmd, prev_dq_raw)
        dq_cmd = adaptive_lowpass(dq_cmd, prev_dq_cmd, dist)
        dq_cmd = rate_limit(dq_cmd, prev_dq_cmd, max_rate=2.0, dt=dt)

        # --- Safety ---
        dq_cmd *= torso_safety_scale(p_hand)
        dq_cmd = soft_clip(dq_cmd, 0.8 * env.action_space.high)

        if not np.isfinite(dq_cmd).all():
            dq_cmd = np.zeros_like(dq_cmd)

        dq_cmd = dq_cmd.astype(np.float32)

        obs, reward, done, truncated, info = env.step(dq_cmd)
        if prev_dq_cmd is None:
            rate = np.zeros_like(dq_cmd)
        else:
            rate = (dq_cmd - prev_dq_cmd) / dt

        rates.append(rate.copy())

        prev_dq_raw = dq_cmd.copy()
        prev_dq_cmd = dq_cmd.copy()

        env.render()
        if done or truncated:
            break

    all_joint_delta_rates.append(np.stack(rates))
    all_final_dists.append(dist)

    print(f"Episode {ep+1}: steps={step+1}, final dist={dist:.4f} m")


# ============================================================
# Diagnostics
# ============================================================

def pad_3d(arrs, pad_val=np.nan):
    n = len(arrs)
    T = max(a.shape[0] for a in arrs)
    D = arrs[0].shape[1]
    out = np.full((n, T, D), pad_val)
    for i, a in enumerate(arrs):
        out[i, :a.shape[0]] = a
    return out


rates_3d = pad_3d(all_joint_delta_rates)
rms_rate = np.sqrt(np.nanmean(rates_3d**2, axis=2))

mean_rms = np.nanmean(rms_rate, axis=0)
std_rms = np.nanstd(rms_rate, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(mean_rms, 'k', lw=2)
plt.fill_between(
    np.arange(len(mean_rms)),
    mean_rms - std_rms,
    mean_rms + std_rms,
    alpha=0.3
)
plt.xlabel("Timestep")
plt.ylabel("RMS d(Δq)/dt (rad/s²)")
plt.title("Command Smoothness (Mean ± Std)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(all_final_dists, 'o-')
plt.xlabel("Episode")
plt.ylabel("Final Distance (m)")
plt.title("Final Distance to Target")
plt.grid(True)
plt.tight_layout()
plt.show()
