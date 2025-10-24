from environment.env import RacingEnv
from environment.track import RaceTrack
from environment.simplecar import SimpleCar
from agents.ppo import PPOAgent
import numpy as np
import os

# --- Hyperparameters ---
num_episodes = 2000
max_steps = 1000
save_dir = "./ppo_checkpoints"
os.makedirs(save_dir, exist_ok=True)
gif_every = 50  # optional

# --- Create environment ---
theta = np.linspace(0, 2*np.pi, 2000)
waypoints = np.stack([50*np.cos(theta), 30*np.sin(theta)], axis=1)
track = RaceTrack(waypoints, trackwidth=10)
car = SimpleCar()
env = RacingEnv(track, car, max_steps=max_steps)

# --- Create PPO agent ---
obs_dim = env.obs_dim
act_dim = env.action_dim
agent = PPOAgent(obs_dim, act_dim, device='cuda')

# --- Training loop ---
for ep in range(num_episodes):
    state = env.reset()
    ep_reward = 0
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.store_reward_done(reward, done)
        state = next_state
        ep_reward += reward
        if done:
            break

    agent.update()

    if ep % 10 == 0:
        print(f"Episode {ep}, Reward: {ep_reward:.2f}")

    # Optional: render GIF
    if gif_every is not None and ep % gif_every == 0:
        env.car_traj = []
        if hasattr(env, "_frames"):
            del env._frames
        s_vis = env.reset()
        done_vis = False
        steps_gif = 0
        render_every = 10
        while not done_vis and steps_gif < max_steps:
            a_vis = agent.act(s_vis)
            ns_vis, r_vis, done_vis, info_vis = env.step(a_vis)
            if steps_gif % render_every == 0:
                env.render(mode="gif")
            s_vis = ns_vis
            steps_gif += 1
        env.save_gif(os.path.join(save_dir, f"ppo_ep{ep:05d}.gif"))
