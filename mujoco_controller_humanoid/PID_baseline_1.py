import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from g1_arm_rl_env import G1ArmReachEnv
import mujoco

def distance_gain(dist, kp_far=2.5, kp_near=0.6, d_far=0.30):
    w = np.clip(dist / d_far, 0.0, 1.0)
    return kp_near + w * (kp_far - kp_near)


def torso_safety_scale(p_hand, torso_pos=np.array([0.0, 0.0, 0.90]), safe_radius=0.25):
    d = np.linalg.norm(p_hand - torso_pos)
    if d >= safe_radius:
        return 1.0
    return np.clip(d / safe_radius, 0.2, 1.0)

def task_space_impedance(p_hand, v_hand, p_goal,
                          Kp=100.0, Kd=20.0):
    return Kp * (p_goal - p_hand) - Kd * v_hand

def damped_jacobian_pinv(J, damping=1e-3):
    JJt = J @ J.T
    return J.T @ np.linalg.inv(JJt + damping * np.eye(JJt.shape[0]))
def nullspace_control(q, q_rest, K_null=0.5):
    return -K_null * (q - q_rest)

def jerk_regularization(delta_q, prev_delta_q, K_jerk=0.2):
    if prev_delta_q is None:
        return delta_q
    return delta_q - K_jerk * (delta_q - prev_delta_q)

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

# def compute_jacobian(env, q, eps=1e-4):
#     for idx, val in zip(env._qadr, q):
#         env.data.qpos[idx] = val
#     env._mujoco.mj_forward(env.model, env.data)

#     p0 = env._fk()
#     J = np.zeros((3, len(q)))

#     for i in range(len(q)):
#         dq = q.copy()
#         dq[i] += eps
#         for idx, val in zip(env._qadr, dq):
#             env.data.qpos[idx] = val
#         env._mujoco.mj_forward(env.model, env.data)
#         pi = env._fk()
#         J[:, i] = (pi - p0) / eps

#     return J

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

# PID controller parameters
Kp = 2.5
Kd = 0.2

def pid_delta_q(q, dq, q_desired):
    return Kp * (q_desired - q) - Kd * dq

# --- Run multiple episodes with rendering ---
N_EPISODES = 150
MAX_STEPS = 500

env = G1ArmReachEnv(render_mode='human')
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
all_dists = []
all_joint_delta_rates = []
all_joint_errors = []
all_steps = []
all_final_dists = []

q_slice = slice(0, 7)
dq_slice = slice(7, 14)
p_hand_slice = slice(14, 17)
p_goal_slice = slice(17, 20)
delta_slice = slice(20, 23)

for episode in range(N_EPISODES):
    obs, _ = env.reset()
    dists = []
    joint_deltas = []
    joint_errors = []
    joint_delta_rates = []
    prev_delta_q = None 
    n_steps = 0
    done = False

    while not done and n_steps < MAX_STEPS:
        q = obs[q_slice]
        dq = obs[dq_slice]
        p_hand = obs[p_hand_slice]
        p_goal = obs[p_goal_slice]
        delta_p = obs[delta_slice]
        dt = env.model.opt.timestep
        dist = np.linalg.norm(p_goal - p_hand)
        # q_desired = numerical_ik(env, q, p_goal, steps=3, alpha=0.8)
        # delta_q = pid_delta_q(q, dq, q_desired)
        # delta_q = np.clip(delta_q, -0.05, 0.05)

        # --- distance-aware PID ---
        # Compute Jacobian J and hand velocity
        # J = compute_jacobian(env, q)          # analytic or numerical
        J = compute_jacobian_point(
            env,
            q,
            body_name="right_wrist_yaw_link",
            point_offset=[0.0415, -0.003, 0.0]
        )
        v_hand = J @ dq

        # Task-space impedance
        F = task_space_impedance(p_hand, v_hand, p_goal)

        # Task-space joint command
        J_pinv = damped_jacobian_pinv(J)
        dq_task = J_pinv @ F

        # Null-space posture control
        dq_null = nullspace_control(q, q_rest=np.zeros_like(q))
        N = np.eye(len(q)) - J_pinv @ J
        dq_cmd = dq_task + N @ dq_null

        # Jerk regularization
        dq_cmd = jerk_regularization(dq_cmd, prev_delta_q)

        # Safety scaling + clipping
        dq_cmd *= torso_safety_scale(p_hand)
        # dq_cmd = np.clip(dq_cmd, -0.05, 0.05)
                # print(delta_q)
        # --- Action safety (MANDATORY) ---
        dq_cmd = np.asarray(dq_cmd, dtype=np.float32)
        dq_cmd = np.clip(dq_cmd, env.action_space.low, env.action_space.high)

        # --- Step ---
        obs, reward, done, truncated, info = env.step(dq_cmd)
        # obs, reward, done, truncated, info = env.step(delta_q)
        # if prev_delta_q is None:
        #     delta_q_rate = np.zeros_like(delta_q)
        # else:
        #     delta_q_rate = (delta_q - prev_delta_q) / dt #delta_q - prev_delta_q   # per-joint rate of change

        if prev_delta_q is None:
            delta_q_rate = np.zeros_like(dq_cmd)
        else:
            delta_q_rate = (dq_cmd - prev_delta_q) / dt

        prev_delta_q = dq_cmd.copy()
        
        joint_delta_rates.append(delta_q_rate.copy())
        
        env.render()

        n_steps += 1
        # print(dist)
    # final_dist = dists[-1]
    all_final_dists.append(dist)
    all_joint_delta_rates.append(np.stack(joint_delta_rates))
    print(f"Episode {episode+1}: steps={n_steps}, final dist={dist:.4f} m")


plt.figure(figsize=(8, 4))
plt.title("Final Distance to Target per Episode")
plt.plot(all_final_dists, 'o-', lw=2)
plt.xlabel("Episode")
plt.ylabel("Final Distance (m)")
plt.grid(True)
plt.tight_layout()
plt.show()

def pad(arrs, pad_val=np.nan):
    maxlen = max(len(a) for a in arrs)
    out = np.full((len(arrs), maxlen), pad_val)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out

# -------------------------------
# Rate-of-change plotting
# -------------------------------
rates_pad = pad([np.linalg.norm(r, axis=1) for r in all_joint_delta_rates])
# rates_pad = pad([np.linalg.norm(r, axis=1) for r in all_joint_delta_rates])

plt.figure(figsize=(10, 4))
plt.title("RMS Rate of Change of Joint Commands")
for i in range(N_EPISODES):
    plt.plot(rates_pad[i], alpha=0.4)
plt.plot(np.nanmean(rates_pad, axis=0), 'k', lw=2)
plt.xlabel("Timestep")
plt.ylabel("‖Δ(Δq)‖")
plt.tight_layout()
plt.show()
def pad_3d(arrs, pad_val=np.nan):
    """
    Pads a list of (T_i, D) arrays into (N, T_max, D)
    """
    n = len(arrs)
    maxlen = max(a.shape[0] for a in arrs)
    dim = arrs[0].shape[1]

    out = np.full((n, maxlen, dim), pad_val)
    for i, a in enumerate(arrs):
        out[i, :a.shape[0], :] = a
    return out



rates_pad = pad_3d(all_joint_delta_rates)


plt.figure(figsize=(14, 10))
for j in range(7):
    plt.subplot(4, 2, j+1)
    for ep in range(rates_pad.shape[0]):
        plt.plot(rates_pad[ep, :, j], alpha=0.3)
    plt.plot(np.nanmean(rates_pad[:, :, j], axis=0), 'k', lw=2)
    plt.title(f"Joint {j+1}: d(Δq)/dt")
    plt.xlabel("Timestep")
    plt.ylabel("rad/s²")

plt.tight_layout()
plt.show()

rms_rate = pad([
    np.linalg.norm(r, axis=1) for r in all_joint_delta_rates
])

plt.figure(figsize=(10, 4))
plt.plot(np.nanmean(rms_rate, axis=0), 'k', lw=2)
plt.xlabel("Timestep")
plt.ylabel("‖d(Δq)/dt‖  (rad/s²)")
plt.title("RMS Rate of Change of Joint Commands")
plt.tight_layout()
plt.show()


mean_rates = np.nanmean(rates_pad, axis=0)  # (T, 7)
std_rates  = np.nanstd(rates_pad, axis=0)   # (T, 7)

plt.figure(figsize=(14, 10))
for j in range(7):
    plt.subplot(4, 2, j+1)

    t = np.arange(mean_rates.shape[0])

    plt.plot(t, mean_rates[:, j], 'k', lw=2, label="Mean")
    plt.fill_between(
        t,
        mean_rates[:, j] - std_rates[:, j],
        mean_rates[:, j] + std_rates[:, j],
        alpha=0.3,
        label="±1 Std"
    )

    plt.title(f"Joint {j+1}: d(Δq)/dt")
    plt.xlabel("Timestep")
    plt.ylabel("rad/s²")

plt.tight_layout()
plt.show()


rms_rate = np.sqrt(np.nanmean(rates_pad**2, axis=2))  # (E, T)
rms_mean = np.nanmean(rms_rate, axis=0)
rms_std  = np.nanstd(rms_rate, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(rms_mean, 'k', lw=2)
plt.fill_between(
    np.arange(len(rms_mean)),
    rms_mean - rms_std,
    rms_mean + rms_std,
    alpha=0.3
)
plt.xlabel("Timestep")
plt.ylabel("RMS d(Δq)/dt (rad/s²)")
plt.title("Command Smoothness (Mean ± Std)")
plt.tight_layout()
plt.show()


