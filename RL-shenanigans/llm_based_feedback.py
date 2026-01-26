import os
import json
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from g1_arm_rl_env import make_env
import openai  # requires `pip install openai`

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# ======================================================
# 1. Define helper: summarize an episode
# ======================================================
def summarize_episode(ep_info):
    """Convert raw episode info into a concise summary for the LLM."""
    summary = f"""
    Episode summary:
    - Total reward: {ep_info['ep_ret']:.2f}
    - Steps: {ep_info['steps']}
    - Final distance to goal: {ep_info['final_dist']:.3f} m
    - Collisions: {ep_info['collisions']}
    - Termination reason: {ep_info['termination']}
    - Avg joint velocity: {ep_info['avg_vel']:.3f} rad/s
    - Avg step reward: {ep_info['ep_ret']/max(ep_info['steps'],1):.3f}
    """
    return summary.strip()


# ======================================================
# 2. Define helper: call LLM to analyze reward
# ======================================================
def query_llm_for_feedback(summary: str) -> dict:
    """Ask the LLM to diagnose and suggest reward adjustments."""
    prompt = f"""
    You are a robotics reinforcement learning expert.

    Analyze the following robot arm reaching episode and explain
    why the total reward might be low. Suggest concrete adjustments
    to reward terms or shaping logic that could improve learning.

    Return your response strictly as JSON with the keys:
    {{
        "diagnosis": "short textual reason",
        "suggested_changes": {{
            "term": "description of what to change"
        }}
    }}

    Episode data:
    {summary}
    """

    response = openai.ChatCompletion.create(
        model="gpt-5",  # or "gpt-4-turbo" if available
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"diagnosis": text, "suggested_changes": {}}


# ======================================================
# 3. Run training with episode-level LLM feedback
# ======================================================
def run_training(num_episodes=10, reward_threshold=-50.0):
    env = make_env()
    model = PPO("MlpPolicy", env, verbose=0)
    log_file = f"llm_feedback_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done, ep_ret, steps = False, 0.0, 0
        total_collisions = 0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, r, terminated, truncated, info = env.step(action)
            ep_ret += r
            steps += 1
            done = terminated or truncated
            if info.get("collision", False):
                total_collisions += 1

        final_dist = info.get("dist", np.nan)
        avg_vel = np.mean(np.abs(env.data.qvel[env._qadr]))

        ep_info = dict(
            ep=ep,
            ep_ret=ep_ret,
            steps=steps,
            final_dist=final_dist,
            collisions=total_collisions,
            termination="goal_reached" if final_dist < 0.03 else "timeout_or_collision",
            avg_vel=avg_vel,
        )

        summary = summarize_episode(ep_info)

        # Query the LLM if episode underperforms
        if ep_ret < reward_threshold:
            feedback = query_llm_for_feedback(summary)
            ep_info["llm_feedback"] = feedback
            print(f"[Episode {ep}] Low reward detected. LLM feedback:")
            print(json.dumps(feedback, indent=2))
        else:
            ep_info["llm_feedback"] = {}

        # Save feedback to log
        with open(log_file, "a") as f:
            f.write(json.dumps(ep_info) + "\n")

        # Continue RL training update
        model.learn(total_timesteps=1000, reset_num_timesteps=False)

    env.close()
    print(f"LLM feedback logged to {log_file}")
    return log_file


if __name__ == "__main__":
    run_training(num_episodes=10)
