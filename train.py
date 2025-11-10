# train.py
from environment.env import RacingEnv
from environment.track import RaceTrack
from environment.simplecar import SimpleCar
from environment.circuit2waypoint import extract_track_from_image
from agents.dqn import DQNAgent
import numpy as np
import time, torch, os

# -------------------------
# Utility: build discrete action set
# -------------------------
def build_discrete_actions(n_throttle=3, n_steer=3,
                           throttle_range=(0.0, 1.0),
                           steer_range=(-1.0, 1.0),
                           include_brake=False):
    """
    Create a list of (throttle, steer) tuples that discretize the continuous controls.

    - throttle_range: (min, max). If include_brake True, min can be negative for braking.
    - steer_range: typical normalized steering [-1, 1]
    Returns: action_list (list of tuples), and an index->tuple mapping
    """
    if include_brake:
        # allow negative throttle (braking)
        th_min, th_max = -abs(throttle_range[1]), throttle_range[1]
    else:
        th_min, th_max = throttle_range

    throttles = np.linspace(th_min, th_max, n_throttle)
    steers = np.linspace(steer_range[0], steer_range[1], n_steer)

    actions = []
    for t in throttles:
        for s in steers:
            actions.append((float(t), float(s)))
    return actions

# -------------------------
# Training loop
# -------------------------
def train_dqn(env, agent, action_map,
              num_episodes=5000,
              max_steps_per_episode=2000,
              save_dir="checkpoints",
              checkpoint_every=200,
              gif_every=None,
              verbose=True):
    """
    env: RacingEnv instance (expects .reset(), .step(action), optional .render(), .save_gif())
    agent: DQNAgent instance. Must provide:
           - act(state) -> discrete action index
           - store(s,a,r,ns,done)
           - learn()
           - (optional) save(path)
           - (optional) epsilon (for logging/decay)
    action_map: list mapping discrete index -> actual continuous action tuple accepted by env.step
    """
    os.makedirs(save_dir, exist_ok=True)

    # best lap time (lower is better). initialize to +inf unless we find lap info differently
    best_lap = float('inf')
    best_reward = -float('inf')
    start_time = time.time()

    # determine state size if agent creation requires it somewhere else.
    # Here we only use states returned by env.reset()
    for ep in range(1, num_episodes + 1):
        s = env.reset()
        # if env.reset returns tuple (state, info) handle it gracefully
        if isinstance(s, tuple):
            s = s[0]

        ep_reward = 0.0
        done = False
        step = 0

        # optional: let env render initial state (only if interactive)
        # env.render()  # uncomment if you want live rendering each episode

        while not done and step < max_steps_per_episode:
            # agent.act expects the observation; it returns an integer index into action_map
            a_idx = agent.act(s)
            # sanity: make sure a_idx is int and within range
            if not isinstance(a_idx, (int, np.integer)) or a_idx < 0 or a_idx >= len(action_map):
                # fallback: if agent returns continuous action directly, try to use it
                if isinstance(a_idx, (tuple, list, np.ndarray)):
                    action = tuple(a_idx)
                else:
                    # fallback to random discrete action
                    a_idx = int(np.random.randint(0, len(action_map)))
                    action = action_map[a_idx]
            else:
                action = action_map[int(a_idx)]

            # if step % 100 == 0:
            #     ns, r, done, info = env.step(action, debug = True)
            ns, r, done, info = env.step(action)
            
            if isinstance(ns, tuple):
                ns = ns[0]

            r = np.clip(r, -50, 50)
            
            agent.store(s, a_idx, r, ns, done)
            agent.learn()

            s = ns
            ep_reward += float(r)
            step += 1

        # episode finished: logging & checkpointing
        # attempt to read lap time or meaningful info keys from env/ info
        if info.get('lap_completed', False):
            lap_time = info['lap_time']
            print(f"Episode {ep} completed a lap in {lap_time:.2f} seconds")
        else:
            lap_time = float('inf')
            print(f"Episode {ep} ended at step {env.step_count}, progress: {env.prev_s}")

 
        if ep_reward > best_reward:
            best_reward = ep_reward

        # Logging
        if verbose and (ep % 10 == 0):
            eps_info = f", eps={getattr(agent, 'epsilon', 'N/A'):.3f}" if hasattr(agent, 'epsilon') else ""
            lap_info = f", lap_time={lap_time:.3f}" if lap_time is not None else ""
            print(f"[{ep}/{num_episodes}] reward={ep_reward:.2f}{lap_info}, best_reward={best_reward:.2f}, best_lap={best_lap if best_lap != float('inf') else 'N/A'}{eps_info}")

        # Save checkpoint
        if ep % checkpoint_every == 0:
            ckpt_name = os.path.join(save_dir, f"dqn_ep{ep:05d}")
            torch.save(agent.net.state_dict(), ckpt_name)
            print(f'Saved model parameters for ep {ep} to: {ckpt_name}')
    
        # Save gifs
        if (gif_every is not None and ep % gif_every == 0) or env.prev_s > 0.85:
            try:
                gif_path = os.path.join(save_dir, f"policy_ep{ep:05d}.gif")
                print(f"Generating GIF for episode {ep}...")

                # Reset the environment and clear trajectory
                s_vis = env.reset()
                env.car_traj = []  # Clear any previous trajectory
                done_vis = False
                steps_gif = 0

                # Clear previous frames if any
                if hasattr(env, "_frames"):
                    del env._frames  # reset previous frame list

                render_every = 20  # only render every 10th step

                while not done_vis and steps_gif < max_steps_per_episode:
                    a_idx = agent.act(s_vis)
                    action = action_map[int(a_idx)] if isinstance(a_idx, (int, np.integer)) else action_map[np.random.randint(len(action_map))]
                    ns_vis, r_vis, done_vis, info_vis = env.step(action)

                    # Render only every N-th step
                    if steps_gif % render_every == 0:
                        env.render(mode="gif")
                    
                    s_vis = ns_vis[0] if isinstance(ns_vis, tuple) else ns_vis
                    steps_gif += 1
                
                env.save_gif(gif_path)
                print(f"Saved GIF for episode {ep}")

            except Exception as e_gif:
                print(f"Failed to generate GIF at episode {ep}: {e_gif}")

            
    total_time = time.time() - start_time
    print(f"Training complete. Episodes: {num_episodes}, elapsed {total_time:.1f}s")

# -------------------------
# Example main: create env, discretize actions, instantiate agent, run training
# -------------------------
if __name__ == "__main__":
    # create track, car, env (adjust arguments to match your constructors)
    # Example: use a simple synthetic track
    # theta = np.linspace(0, 2*np.pi, 2000)
    # waypoints = np.stack([50*np.cos(theta), 30*np.sin(theta)], axis=1)
    # track = RaceTrack(waypoints, trackwidth=10)

    # Silverstone
    waypoints = extract_track_from_image(
        "environment/tracks/saopaulo.png",
        invert=True,
        plot_steps=False
    )
    print('Extracted waypoints')
    track = RaceTrack(waypoints, name="Sao Paulo", trackwidth=15)

    car = SimpleCar(init_pos=track.centerline[0])
    env = RacingEnv(track=track, car=car)

    # build discrete action mapping (throttle x steering)
    action_map = build_discrete_actions(n_throttle=3, n_steer=3,
                                        throttle_range=(-1.0, 2.5),
                                        steer_range=(-1, 1),
                                        include_brake=True)
    action_size = len(action_map)

    # infer state size from env if possible
    try:
        obs_space = env.observation_space
        # gym spaces often have shape attribute
        state_size = int(np.prod(obs_space.shape))
    except Exception:
        # fallback to probing a reset() observation
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        state_size = int(np.size(obs))

    # instantiate your DQNAgent (constructor args may vary; adapt as needed)
    # Typical signature (state_size, action_size, **kwargs)
    agent = DQNAgent(obs_dim=state_size, n_actions=action_size, dict_path= '/mnt/cloudNAS3/arpita/Projects/rleclerc/dqn_checkpoints/dqn_ep02500')
    print(f'Using: {agent.device}')
    # run training
    train_dqn(env, agent,
              action_map,
              num_episodes=3000,
              max_steps_per_episode=4000,
              save_dir="dqn_checkpoints",
              checkpoint_every=100,
              gif_every=100,
              verbose=True)
