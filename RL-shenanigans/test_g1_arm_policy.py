#!/usr/bin/env python3.10

from g1_arm_rl_CGP_env import G1ArmReachEnv
import gymnasium as gym
from stable_baselines3 import PPO
import os
from stable_baselines3 import PPO
import numpy as np 

import matplotlib.pyplot as plt
# def test_pretrained_policy(
#     checkpoint: str,
#     num_episodes: int = 20,
#     right_arm: bool = False,
#     headless: bool = False,
#     use_cpg: bool = True,
# ):
#     assert os.path.isfile(checkpoint), f"Checkpoint not found: {checkpoint}"

#     env = G1ArmReachEnv(
#         render_mode=None if headless else "human",
#         right_arm=right_arm,
#     )

#     env._use_cpg = use_cpg

#     model = PPO.load(checkpoint, env=env, device="auto")

#     for ep in range(num_episodes):
#         obs, _ = env.reset()
#         done = False
#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, _, done, _, _ = env.step(action)

#     env.close()
# if __name__ == "__main__":
#     test_pretrained_policy(
#         checkpoint="/home/suleman/unitree_rl_gym/unitree_g1_vibes/RL-shenanigans/models/ppo_g1_left_53178k.zip")

all_q = []
all_dq = []
all_actions = []
all_cpg_actions = []
import numpy as np


class JointPID:
    """
    Residual joint-space PID used as a *stabilizer* on top of RL Δq actions.

    This PID:
    - Tracks a moving joint target q_cmd (integrated RL output)
    - Produces a *small* Δq correction (not torque)
    - Uses velocity for derivative (stable, low-noise)
    - Has bounded integral and bounded output
    """

    def __init__(
        self,
        kp,
        kd,
        ki=0.0,
        action_dim=7,
        i_limit=0.1,
        output_limit=0.02,
    ):
        self.kp = np.asarray(kp, dtype=np.float32)
        self.kd = np.asarray(kd, dtype=np.float32)
        self.ki = np.asarray(ki, dtype=np.float32)

        self.integral = np.zeros(action_dim, dtype=np.float32)

        self.i_limit = float(i_limit)
        self.output_limit = float(output_limit)

    def reset(self):
        self.integral[:] = 0.0

    def compute(self, q, q_cmd, qd, dt):
        """
        Args:
            q     : current joint positions   [rad]
            q_cmd : desired joint positions   [rad]
            qd    : joint velocities          [rad/s]
            dt    : timestep                  [s]

        Returns:
            delta_q_pid : small stabilizing Δq correction
        """
        # Position error
        error = q_cmd - q

        # Integral (bounded)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.i_limit, self.i_limit)

        # Derivative term (velocity-based, NOT finite-difference)
        d_error = -qd

        # PID output (Δq correction)
        delta = (
            self.kp * error
            + self.kd * d_error
            + self.ki * self.integral
        )

        # Important: PID is a *small residual*, not the main action
        delta = np.clip(delta, -self.output_limit, self.output_limit)
        return delta

pid = JointPID(
    kp=[0.6, 0.6, 0.5, 0.5, 0.3, 0.3, 0.2],
    kd=[0.08, 0.08, 0.06, 0.06, 0.04, 0.04, 0.03],
    ki=0.0,
    action_dim=7,
    output_limit=0.02,   # smaller than RL Δq
)



def test_pretrained_policy(
    checkpoint: str = "/home/suleman/unitree_rl_gym/unitree_g1_vibes/RL-shenanigans/models/ppo_g1_left_53178k.zip",
    num_episodes: int = 20,
    right_arm: bool = False,
    headless: bool = False,
    use_cpg: bool = True,
) -> None:
    """
    Load a pretrained PPO policy and evaluate it without training.

    Args:
        checkpoint: Path to SB3 .zip model
        num_episodes: Number of evaluation episodes
        right_arm: Whether to test right or left arm
        headless: Disable viewer
        use_cpg: Toggle RL vs RL+CPG
    """

    from stable_baselines3 import PPO

    # assert checkpoint.is_file(), f"Checkpoint not found: {checkpoint}"

    # ------------------------------------------------------------
    # Create a single evaluation environment (NO VecEnv needed)
    # ------------------------------------------------------------
    env = G1ArmReachEnv(
        render_mode=None if headless else "human",
        right_arm=right_arm,
    )
    # pid = JointPID(kp=1.0, ki=0.0, kd=0.1, action_dim=7)

    # Toggle CPG explicitly (important for ablations)
    env._use_cpg = use_cpg

    # ------------------------------------------------------------
    # Load model (policy weights only; no optimizer needed)
    # ------------------------------------------------------------
    model = PPO.load(checkpoint, env=env, device="auto")

    print(f"\nLoaded checkpoint: {checkpoint}")
    print(f"Evaluation mode: {'RL+CPG' if use_cpg else 'RL-only'}")
    print(f"Arm: {'right' if right_arm else 'left'}")
    print(f"Episodes: {num_episodes}\n")

    # ------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------
    success_count = 0
    total_return = 0.0
    min_torso_dists = []
    action_norms = []
    cpg_action_norms = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_min_torso = float("inf")
        # safe_q = env.data.qpos[env._qadr] 
        pid.reset()

       
        # q_cmd = None  

        
        

        while not done:
            # current_q = env.data.qpos[env._qadr]
            action, _ = model.predict(obs, deterministic=True)
            #action = env.action_space.sample() #model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            ep_return += reward

            # Track safety-related quantities
            p_hand = env._fk()
            torso_pos = np.array([0.0, 0.0, 0.90])
            ep_min_torso = min(ep_min_torso, np.linalg.norm(p_hand - torso_pos))

            if "action_l2" in info:
                action_norms.append(info["action_l2"])
            if "cpg_action_l2" in info:
                cpg_action_norms.append(info["cpg_action_l2"])
            # Log joint positions and velocities
            all_q.append(env.data.qpos[env._qadr].copy())
            all_dq.append(env.data.qvel[env._qadr].copy())

            # Log actions
            all_actions.append(action.copy())

            if env._use_cpg:
                dq_cpg = env._cpg.step(action)
                all_cpg_actions.append(dq_cpg.copy())

        total_return += ep_return
        min_torso_dists.append(ep_min_torso)

        if info.get("dist", 1.0) < 0.03:
            success_count += 1

        print(
            f"[Episode {ep+1:02d}] "
            f"Return={ep_return:7.2f} | "
            f"Min torso dist={ep_min_torso*100:5.1f} cm"
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n=== Evaluation Summary ===")
    print(f"Success rate: {success_count}/{num_episodes}")
    print(f"Mean return: {total_return / num_episodes:.2f}")
    print(f"Mean min torso distance: {np.mean(min_torso_dists)*100:.1f} cm")

    if action_norms:
        print(f"Mean RL action L2: {np.mean(action_norms):.3f}")
    if cpg_action_norms:
        print(f"Mean CPG action L2: {np.mean(cpg_action_norms):.3f}")
    Q = np.array(all_q)              # shape [T, 7]
    DQ = np.array(all_dq)            # shape [T, 7]
    A = np.array(all_actions)        # shape [T, 7]
    A_cpg = np.array(all_cpg_actions)
    t = np.arange(Q.shape[0]) * env.model.opt.timestep
    plt.figure(figsize=(10, 6))
    for j in range(7):
        plt.plot(t, Q[:, j], label=f"Joint {j}")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint angle [rad]")
    plt.title("Joint Position Trajectories")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
    for j in range(7):
        plt.plot(t[:-1], np.diff(Q[:, j]) / env.model.opt.timestep)
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity [rad/s]")
    plt.title("Joint Velocities")
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(8, 4))
    plt.plot(t, np.linalg.norm(A, axis=1), label="RL action ||Δq||")
    plt.plot(t, np.linalg.norm(A_cpg, axis=1), label="CPG-filtered ||Δq||")
    plt.xlabel("Time [s]")
    plt.ylabel("Action magnitude")
    plt.title("Action Magnitude Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
    dt = env.model.opt.timestep
    dq = np.diff(Q, axis=0) / dt 
    ddq = np.diff(dq, axis=0) / dt      # shape (T-2, 7)
    jerk = np.diff(ddq, axis=0) / dt    # shape (T-3, 7)
    jerk_mag = np.linalg.norm(jerk, axis=1)  # shape (T-3,)
    t_jerk = np.arange(len(jerk_mag)) * dt





    plt.figure(figsize=(8, 4))
    plt.plot(t_jerk, jerk_mag)
    plt.xlabel("Time [s]")
    plt.ylabel("Joint jerk [rad/s³]")
    plt.title("Joint-space jerk magnitude")
    plt.grid(True)
    plt.show()




    env.close()

if __name__ == "__main__":
    test_pretrained_policy()
    
