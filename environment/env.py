'''
Environment wrapper for race simulator gym
'''

import numpy as np
import matplotlib.pyplot as plt
import os, imageio

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

    def step(self, action):
        '''
        Step the environment given action 
        '''
        throttle, steer = action
        self.car.step(throttle, steer)
        self.step_count += 1

        obs = self._get_obs() # Get state from action
        reward = self._compute_reward(obs)
        done = self._check_done(obs)

        info = {'s_progress': obs[4]}
        return obs, reward, done, info
    
    def _compute_reward(self, obs):
        '''
        Reward funciton that:
            + progress along track
            + alignment with track direction
            + speed maintenance
            - penalty if off track
        '''

        v_norm, heading_err, lat_err, curvature, progress = obs
        progress_delta = progress - self.prev_s
        self.prev_s = progress

        # Reward terms
        reward_progress = 200.0 * max(progress_delta, 0)
        reward_align = np.cos(heading_err)
        reward_speed = 0.5 * v_norm

        # Combine terms
        reward = reward_progress * reward_align + reward_speed

        # Off_track penalty
        if abs(lat_err) > 1.0:
            reward -= 10.0
        
        return reward
    
    def _check_done(self, obs):
        '''
        Terminate if off track or max steps
        '''
        _, lat_err, _, _, _ = obs
        if abs(lat_err) > 1.0: return True 
        if self.step_count >= self.max_steps: return True
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
        for f in self._frames:
            os.remove(f)
        os.rmdir(self._tmp_dir)
        del self._frames


        