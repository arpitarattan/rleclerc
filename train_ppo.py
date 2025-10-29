from environment.env import RacingEnv
from environment.track import RaceTrack
from environment.simplecar import SimpleCar
from agents.ppo import PPOAgent
from collections import deque
import numpy as np
import os, torch
from environment.circuit2waypoint import extract_track_from_image

# --- Hyperparameters ---

num_episodes = 2000
max_steps = 2000
save_dir = "./ppo_checkpoints"
save_every = 100  # save every N episodes
os.makedirs(save_dir, exist_ok=True)
gif_every = 20  # optional

# --- Create environment ---

# theta = np.linspace(0, 2 * np.pi, 2000)
# waypoints = np.stack([50 * np.cos(theta), 30 * np.sin(theta)], axis=1)
# track = RaceTrack(waypoints, trackwidth=10)

# Silverstone
waypoints = extract_track_from_image(
    "environment/tracks/silverstone.jpeg",
    invert=True,
    plot_steps=False
)
print('Extracted waypoints')
track = RaceTrack(waypoints, name="Silverstone", trackwidth=15)

car = SimpleCar(init_pos=track.centerline[0])
env = RacingEnv(track, car, max_steps=max_steps)

# --- Create PPO agent ---

obs_dim = env.obs_dim
act_dim = env.action_dim
agent = PPOAgent(obs_dim, act_dim, device='cuda')

# print('Using Pretrained Model...')
# agent.load('./ppo_checkpoints/ppo_ep00500.pt', map_location='cuda')
lap_time_window = deque(maxlen=20)  # rolling average for smoother stats

# --- Training loop ---
rewards = np.array([])
for ep in range(num_episodes):
    state = env.reset()
    car.x += np.random.uniform(-0.1,0.1)
    car.y += np.random.uniform(-0.1,0.1)
    ep_reward = 0
    
    laps_this_ep = 0
    lap_times_this_ep = []

    for step in range(max_steps):
        # --- act and collect rollout data ---
        action_clipped, logprob, value = agent.act(state, return_details=True)
        next_state, reward, done, info = env.step(action_clipped)

        # store full transition
        agent.store_transition(
            torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0),
            torch.tensor(action_clipped, dtype=torch.float32, device=agent.device).unsqueeze(0),
            logprob, value, reward, done
        )
        
        state = next_state
        ep_reward += reward
        rewards = np.append(rewards, ep_reward)

        # --- Log lap completions ---
        if info.get("lap_completed", False):
            laps_this_ep += 1
            if len(info["lap_times"]) > 0:
                lap_time = info["lap_times"][-1]
                lap_times_this_ep.append(lap_time)
                lap_time_window.append(lap_time)
                #print(f"Episode {ep} | Lap {laps_this_ep} completed in {lap_time:.2f}s")

    # --- End of episode ---
    mean_lap_time = np.mean(lap_times_this_ep) if lap_times_this_ep else np.nan
    avg_recent_laps = np.mean(lap_time_window) if len(lap_time_window) > 0 else np.nan

    print(f"EP {ep:4d} | Reward: {ep_reward:8.2f} ")# Laps: {laps_this_ep:2d} | "
         # f"Mean Lap: {mean_lap_time:6.2f}s | Rolling Avg: {avg_recent_laps:6.2f}s")

    agent.update()       

    if ep % save_every == 0:
        model_save_path = os.path.join(save_dir, f"ppo_ep{ep:05d}.pt")
        agent.save(model_save_path)

        rewards_save_path = os.path.join(save_dir, f"ppo_ep{ep:05d}.npy")
        np.save(rewards_save_path, rewards)
        
    # --- Optional: render GIF ---
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
            ns_vis, r_vis, done_vis, info_vis = env.step(a_vis, debug = True)
            if steps_gif % render_every == 0:
                env.render(mode="gif")
            s_vis = ns_vis
            steps_gif += 1
        env.save_gif(os.path.join(save_dir, f"ppo_ep{ep:05d}.gif"))
