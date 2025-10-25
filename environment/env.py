'''
Environment wrapper for race simulator gym
'''

import numpy as np
import matplotlib.pyplot as plt
import os, imageio, shutil

class RacingEnv:
    '''
    Gym wrapper for track + car models
    Observation: state vector
    Action: Can be discrete or continuous actions on throttle + steer
    '''
    def __init__(self, track, car, dt = 0.1, max_steps = 2000):
        self.track = track
        self.car = car
        self.dt = dt
        self.max_steps = max_steps

        self.prev_s = 0.0
        self.step_count = 0

        # observation and action dimensions (for RL)
        self.action_dim = 2  # [throttle, steer]
        self.obs_dim = 5     # [v, lateral_error, heading_error, curvature, progress]

        self.fig, self.ax = None, None # For visualizations

    def reset(self, start_s = 0.0):
        '''
        Reset car to starting line
        '''
        start_pos = self.track.centerline[0]
        tangent = self.track.centerline[1] - self.track.centerline[0]
        init_yaw = np.arctan2(tangent[1], tangent[0])

        self.car.reset(start_pos, yaw=init_yaw)
        self.prev_s = 0.0
        self.step_count = 0
        self.lap_completed = False
        self.prev_progress = 0.0
        self.lap_start_time = 0.0  # start lap timer
        self.current_lap_time = 0.0
        self.lap_times = []
        self.lap_start_step = 0

        obs = self._get_obs()
        return obs
    
    def _get_obs(self):
        '''
        Compute car state relative to track
        '''
        car_pos = np.array([self.car.x, self.car.y])
        track_pts = self.track.centerline

        # Neaerest point on track
        dists = np.linalg.norm(track_pts - car_pos, axis = 1)
        nearest_idx = np.argmin(dists)

        nearest_pt = track_pts[nearest_idx]
        tangent = np.gradient(track_pts, axis = 0)[nearest_idx]
        tangent /= np.linalg.norm(tangent)

        s_progress = self.track.s[nearest_idx]
        track_yaw = np.arctan2(tangent[1], tangent[0])
        heading_error = self._wrap_angle(self.car.yaw - track_yaw)
        lateral_error = np.cross(tangent, car_pos - nearest_pt)
        curvature = self.track.curvature[nearest_idx]

        obs = np.array([
            self.car.v / self.car.max_speed, # Normalized speed
            heading_error,
            lateral_error / (self.track.trackwidth / 2),
            curvature, 
            s_progress / self.track.track_length
        ])
        return obs

    def step(self, action, debug=False):
        '''
        Step the environment given action 
        '''
        throttle, steer = action
        self.car.step(throttle, steer)
        self.step_count += 1

        # update lap time
        self.current_lap_time = self.step_count * self.dt

        obs = self._get_obs() # Get state from action

        # --- Lap completion check ---
        progress = obs[4]  # normalized progress ∈ [0, 1)
        delta_s = progress - self.prev_s

        # Detect wraparound (progress jumped from near 1 → near 0)
        if delta_s < -0.5:
            self.lap_completed = True
            self.lap_times.append(self.current_lap_time)
            print(f"[LAP COMPLETE] Lap {len(self.lap_times)} in {self.current_lap_time:.2f} s")
            self.current_lap_time = 0.0  # reset lap timer
            self.lap_start_step = self.step_count

        self.prev_s = progress

        self.done = False
        reward = self._compute_reward(obs, debug)

        if self.step_count == 1: reward = max(reward, -1.0)
        done = self._check_done(obs)

        info = {'s_progress': obs[4],
            'lap_time': self.current_lap_time,
            'lap_completed': self.lap_completed,
            "lap_times": self.lap_times}
        
        return obs, reward, done | self.done , info
    
    def _compute_reward(self, obs, debug):
        """
        Compute reward for the current car state.
        Combines:
        - forward progress along track
        - speed incentive
        - alignment with track heading
        - lateral error penalty
        - distance-based off-track penalty
        - optional lap completion bonus
        """
        # unpack observation
        v_norm, heading_err, lat_err, curvature, progress = obs

        # -------------------------
        # 1. Track progress reward
        # -------------------------
        delta_m = (progress - self.prev_s) * self.track.track_length
        # handle wrap-around if car loops over s=0
        if delta_m < -0.5 * self.track.track_length:
            delta_m += self.track.track_length
        self.prev_s = progress
        reward_progress = 50.0 * delta_m  # reward per meter

        # -------------------------
        # 2. Speed reward
        # -------------------------
        reward_speed = 30.0 * v_norm + 5.0 * v_norm**2  # nonlinear incentive for faster speed
        if v_norm < 0.05:   # almost stationary
            reward_speed -= 5  # small negative reward
        # -------------------------
        # 3. Heading alignment
        # -------------------------
        reward_heading = 1.0 * np.cos(heading_err)

        # -------------------------
        # 4. Lateral error penalty
        # -------------------------
        reward_lateral = -2.0 * np.clip(abs(lat_err), 0, 1.0)

        # -------------------------
        # 5. True distance-based off-track penalty
        # -------------------------
        car_pos = np.array([self.car.x, self.car.y])
        track_pts = self.track.centerline
        nearest_idx = np.argmin(np.linalg.norm(track_pts - car_pos, axis=1))
        nearest_pt = track_pts[nearest_idx]
        self.dist_offtrack = np.linalg.norm(car_pos - nearest_pt)

        offtrack_penalty = 0.0
        if self.dist_offtrack > self.track.trackwidth / 2:
            offtrack_penalty = -50.0  # strong penalty
            # optionally terminate episode early
            if self.dist_offtrack > self.track.trackwidth:
                self.done = True

        # -------------------------
        # 6. Optional lap completion bonus
        # -------------------------
        lap_bonus = 0.0
        if self.lap_completed and len(self.lap_times) == 1:  # new lap just completed
            lap_bonus = 1000.0

        # -------------------------
        # 7. Combine all components
        # -------------------------
        reward = reward_progress + reward_speed + reward_heading + reward_lateral + offtrack_penalty + lap_bonus

        # Clip reward to avoid TD explosion
        reward = np.clip(reward, -50, 50)

        if debug: 
            # Debug logging (optional)
            print(f"[REW DEBUG] Δm={delta_m:.4f}, prog={progress:.3f}, speed={v_norm:.3f}, "
                f"head={np.cos(heading_err):.3f}, lat={lat_err:.3f}, dist_off={self.dist_offtrack:.3f}, "
                f"off={offtrack_penalty:.3f} -> total={reward:.3f}")

        return reward



    def _check_done(self, obs):
        """
        Terminate if:
        - off-track
        - max steps
        - optional: completed a full lap
        """
        _, lat_err, _, _, _ = obs
        
        # Off-track
        if abs(lat_err) > self.track.trackwidth / 2:
            return True

        # Max steps
        if self.step_count >= self.max_steps:
            return True

        return False

    
    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def render(self, mode="gif", filename="race.gif", fps=30, pause=0.001):
        """
        Renders the racing environment.
        - mode='realtime': shows interactive Matplotlib window
        - mode='gif': saves PNG frames for later compilation
        """
        if not hasattr(self, "car_traj"):
            self.car_traj = []

        # Append current position to trajectory
        self.car_traj.append([self.car.x, self.car.y])

        if mode == "realtime":
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 6))
                self.track.visualize(ax=self.ax, show_curvature=False)
                self.car_marker, = self.ax.plot([], [], 'ro', markersize=5)
                self.traj_line, = self.ax.plot([], [], 'orange', lw=2, alpha=0.7)
                plt.ion()
                plt.show(block=False)

            # Update data
            traj = np.array(self.car_traj)
            self.car_marker.set_data([self.car.x], [self.car.y])
            self.traj_line.set_data(traj[:, 0], traj[:, 1])
            plt.pause(pause)

        elif mode == "gif":
            # Lazy init for GIF rendering
            if not hasattr(self, "_frames"):
                self._frames = []
                self._tmp_dir = "_frames_tmp"
                os.makedirs(self._tmp_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            self.track.visualize(ax=ax,
                                show_curvature=False,
                                car_state=(self.car.x, self.car.y, self.car.yaw),
                                car_traj=self.car_traj)
            ax.set_title(f"Step: {self.step_count}")
            frame_path = os.path.join(self._tmp_dir, f"frame_{self.step_count:05d}.png")
            plt.savefig(frame_path, dpi=120)
            plt.close(fig)
            self._frames.append(frame_path)

        else:
            raise ValueError("mode must be either 'realtime' or 'gif'")

    def save_gif(self, filename="race.gif", fps=30):
        """Combine saved frames into an animated GIF."""
        if not hasattr(self, "_frames"):
            print("No frames to compile.")
            return

        frames = [imageio.imread(f) for f in sorted(self._frames)]
        imageio.mimsave(filename, frames, fps=fps)
        print(f"Saved animation to {filename}")

        # Cleanup
        if hasattr(self, "_tmp_dir") and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir)

        del self._frames


        