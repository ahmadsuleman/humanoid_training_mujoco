import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from g1_arm_rl_env import G1ArmReachEnv

# ============================================================
# Utility functions
# ============================================================

def distance_gain(dist, kp_far=2.5, kp_near=0.6, d_far=0.30):
    w = np.clip(dist / d_far, 0.0, 1.0)
    return kp_near + w * (kp_far - kp_near)

def torso_safety_scale(p_hand, torso_pos=np.array([0.0, 0.0, 0.90]), safe_radius=0.25):
    d = np.linalg.norm(p_hand - torso_pos)
    if d >= safe_radius:
        return 1.0
    return np.clip(d / safe_radius, 0.2, 1.0)

def numerical_ik(env, q_init, p_goal, steps=10, alpha=0.8):
    q = q_init.copy()
    for _ in range(steps):
        for idx, val in zip(env._qadr, q):
            env.data.qpos[idx] = val
        env._mujoco.mj_forward(env.model, env.data)

        p_hand = env._fk()
        delta = p_goal - p_hand
        if np.linalg.norm(delta) < 1e-4:
            break

        J = []
        for i in range(len(q)):
            dq = q.copy()
            dq[i] += 1e-4
            for idx, val in zip(env._qadr, dq):
                env.data.qpos[idx] = val
            env._mujoco.mj_forward(env.model, env.data)
            p_new = env._fk()
            J.append((p_new - p_hand) / 1e-4)

        J = np.stack(J, axis=1)
        dq = alpha * np.linalg.pinv(J) @ delta
        q += dq

    return q

# ============================================================
# PID parameters
# ============================================================

Kd = 0.2

# ============================================================
# Experiment parameters
# ============================================================

N_EPISODES = 30
MAX_STEPS = 1000

env = G1ArmReachEnv(render_mode="human")

# ============================================================
# Data storage
# ============================================================

all_dists = []
all_joint_deltas = []
all_joint_errors = []
all_p_hands = []
all_scales = []

# Observation layout
q_slice = slice(0, 7)
dq_slice = slice(7, 14)
p_hand_slice = slice(14, 17)
p_goal_slice = slice(17, 20)
delta_slice = slice(20, 23)

# ============================================================
# Rollouts
# ============================================================

for ep in range(N_EPISODES):
    obs, _ = env.reset()

    dists = []
    joint_deltas = []
    joint_errors = []
    p_hands = []
    scales = []

    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        q = obs[q_slice]
        dq = obs[dq_slice]
        p_hand = obs[p_hand_slice]
        p_goal = obs[p_goal_slice]
        delta_p = obs[delta_slice]

        # Distance-aware gain
        dist = np.linalg.norm(p_goal - p_hand)
        Kp_eff = distance_gain(dist)

        # IK
        q_desired = numerical_ik(env, q, p_goal, steps=3, alpha=0.8)

        # PID control
        delta_q = Kp_eff * (q_desired - q) - Kd * dq

        # Safety scaling
        scale = torso_safety_scale(p_hand)
        delta_q *= scale

        # Clip
        delta_q = np.clip(delta_q, -0.05, 0.05)

        obs, reward, done, truncated, info = env.step(delta_q)
        env.render()

        # Logging
        dists.append(np.linalg.norm(delta_p))
        joint_deltas.append(delta_q.copy())
        joint_errors.append(q_desired - q)
        p_hands.append(p_hand.copy())
        scales.append(scale)

        steps += 1

    all_dists.append(dists)
    all_joint_deltas.append(np.stack(joint_deltas))
    all_joint_errors.append(np.stack(joint_errors))
    all_p_hands.append(np.stack(p_hands))
    all_scales.append(np.array(scales))

    print(f"Episode {ep+1}: steps={steps}, final dist={dists[-1]:.4f} m")

# ============================================================
# Padding helper
# ============================================================

def pad(arrs, pad_val=np.nan):
    maxlen = max(len(a) for a in arrs)
    out = np.full((len(arrs), maxlen), pad_val)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out

# ============================================================
# Derived metrics
# ============================================================

dists_pad = pad(all_dists)
joint_error_norms = pad([np.linalg.norm(e, axis=1) for e in all_joint_errors])
joint_delta_norms = pad([np.linalg.norm(d, axis=1) for d in all_joint_deltas])
ee_vels = pad([np.linalg.norm(np.diff(p, axis=0), axis=1) for p in all_p_hands])
scales_pad = pad(all_scales, pad_val=1.0)

# ============================================================
# PLOTS
# ============================================================

# 1. Distance to goal
plt.figure(figsize=(12, 5))
plt.title("End-Effector Distance to Goal")
for i in range(N_EPISODES):
    plt.plot(dists_pad[i], alpha=0.4)
plt.plot(np.nanmean(dists_pad, axis=0), "k", lw=2, label="Mean")
plt.xlabel("Timestep")
plt.ylabel("Distance (m)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Joint error norm
plt.figure(figsize=(12, 5))
plt.title("Joint Error Norm ‖q_des − q‖")
for i in range(N_EPISODES):
    plt.plot(joint_error_norms[i], alpha=0.4)
plt.plot(np.nanmean(joint_error_norms, axis=0), "k", lw=2)
plt.xlabel("Timestep")
plt.ylabel("Joint Error Norm")
plt.tight_layout()
plt.show()

# 3. Joint command magnitude
plt.figure(figsize=(12, 5))
plt.title("Joint Command Norm ‖Δq‖")
for i in range(N_EPISODES):
    plt.plot(joint_delta_norms[i], alpha=0.4)
plt.plot(np.nanmean(joint_delta_norms, axis=0), "k", lw=2)
plt.axhline(0.05 * np.sqrt(7), color="r", ls="--", label="Clip limit")
plt.xlabel("Timestep")
plt.ylabel("‖Δq‖")
plt.legend()
plt.tight_layout()
plt.show()

# 4. End-effector velocity
plt.figure(figsize=(12, 5))
plt.title("End-Effector Velocity Magnitude")
for i in range(N_EPISODES):
    plt.plot(ee_vels[i], alpha=0.4)
plt.plot(np.nanmean(ee_vels, axis=0), "k", lw=2)
plt.xlabel("Timestep")
plt.ylabel("‖ẋ_hand‖ (m/step)")
plt.tight_layout()
plt.show()

# 5. Phase plot (distance vs distance rate)
plt.figure(figsize=(6, 6))
plt.title("Phase Plot: Distance Error vs Error Rate")
for d in all_dists:
    e = np.array(d)
    de = np.diff(e)
    plt.plot(e[:-1], de, alpha=0.4)
plt.axhline(0, color="k", lw=0.5)
plt.axvline(0, color="k", lw=0.5)
plt.xlabel("‖p_goal − p_hand‖")
plt.ylabel("Δ‖p_goal − p_hand‖")
plt.tight_layout()
plt.show()

# 6. Safety scaling engagement
plt.figure(figsize=(12, 4))
plt.title("Torso Safety Scale Engagement")
for i in range(N_EPISODES):
    plt.plot(scales_pad[i], alpha=0.4)
plt.plot(np.nanmean(scales_pad, axis=0), "k", lw=2)
plt.xlabel("Timestep")
plt.ylabel("Safety Scale")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
