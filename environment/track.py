'''
Script to define racetrack for RL Simulation of racecar
'''
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

class RaceTrack():
    def __init__(self, waypoints, name='custom', trackwidth = 10.0, step = 0.5, smooth = True):
        self.name = name
        self.trackwidth = trackwidth
        self.raw_waypoints = np.asarray(waypoints) # list of coarse waypoints to contruct track centerline with

        # Precompute track metrics
        self.centerline = self._resample(self.raw_waypoints, step, smooth) 
        self._compute_arclength() # Compute s & length of circuit
        self._compute_tangent_curves()  # Compute headings at each point
        self._compute_boundaries()  # Compute boundary curves of circuit

    # Track geometry preprocessing
    def _resample(self, pts, step = 5, smooth=True):
        '''
        Interpolate coarse waypoints into smooth track
        '''
        # Make sure circuit is closed
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])

        # Compute arclengths from coarse waypoints
        dists = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2 , axis=1))
        s = np.concatenate(([0], np.cumsum(dists)))
        total_length = s[-1] 

        # Interpolate smooth centerline from coarse s
        s_new = np.arange(0, total_length, step)
        if smooth:
            smooth_factor = total_length * 0.02
            tck, u = splprep(pts.T, u=s / total_length, s= smooth_factor, per=True)
            x_new, y_new = splev(np.linspace(0, 1, len(s_new)), tck)
        else:
            x_new = np.interp(s_new, s, pts[:,0])
            y_new = np.interp(s_new, s, pts[:,1])

        return np.stack([x_new, y_new], axis=1)

    def _compute_arclength(self):
        '''
        Compute cumulative distance along centerline
        Creates: 
            self.s: list of arclengths along centerline
            self.track_length: total track length
        '''
        d = np.sqrt(np.sum(np.diff(self.centerline, axis=0) ** 2, axis = 1))
        self.s = np.concatenate(([0], np.cumsum(d))) # Compute arc length from cumulative distances along track
        self.track_length = self.s[-1]

    def _compute_tangent_curves(self):
        '''
        Computes track headings (tangents) at each s along centerline & curvatures for speed management
        '''
        dx, dy = np.gradient(self.centerline[:,0]), np.gradient(self.centerline[:,1]) # Caclulate first order gradients along centerline (x,y)
        d2x, d2y = np.gradient(dx), np.gradient(dy) # Second order grads

        self.headings = np.arctan2(dy, dx)
        self.curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2) ** 1.5

    def _compute_boundaries(self):
        '''
        Compute track boundaries given centerline and width for off-track detection
        '''
        n = np.stack([-np.sin(self.headings), np.cos(self.headings)], axis = 1)
        offset = (self.trackwidth / 2.0) * n
        self.left_boundary = self.centerline + offset
        self.right_boundary = self.centerline - offset

    # Public class methods
    def xy_from_s(self, s):
        '''
        Given progress coord s along centerline, return (x,y, heading)
        '''
        s = s % self.track_length # Local point s
        x, y = np.interp(s, self.s, self.centerline[:,0]), np.interp(s, self.s, self.centerline[:,1])
        heading = np.interp(s, self.s, self.headings)

        return x, y, heading

    def nearest_point(self, x, y):
        '''
        Given global coordinate (x,y) find nearest point on centerline, get:
            - s: progress at that point
            - d: lateral offset from centerline
            - track heading
        '''
        pts = self.centerline
        dists = np.sqrt((pts[:,0]-x)**2 + (pts[:,1] - y) ** 2) # sqrt( (center_x - x)^2 + (center_y - y)^2 ) where (x,y) is car position
        idx = np.argmin(dists)

        s_nearest = self.s[idx]
        x_c, y_c = pts[idx]
        heading = self.headings[idx]

        dx, dy = x - x_c, y - y_c
        lateral = np.sin(heading) * dx - np.cos(heading)*dy # Lateral offset d

        return s_nearest, lateral, heading

    def is_offtrack(self, x, y):
        '''
        Check if car is outside track boundaries
        '''
        _, lateral, _ = self.nearest_point(x,y)
        return abs(lateral) > self.trackwidth / 2.0

    def curvature_at(self, s):
        '''
        Return local curvature at progress s for speed management (slow at turns)
        '''
        s = s % self.total_length
        return np.interp(s, self.s, self.curvature)

    def visualize(self, show_curvature=False, car_state=None, ax=None, car_traj=None):
        """
        Display the racetrack and optionally:
        - color centerline by curvature
        - show car state (x, y, heading)
        - overlay trajectory (car_traj: list of (x, y))
        """
        # Use existing axis if provided, else make new one
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Draw track centerline
        if show_curvature:
            c = self.curvature / (np.max(np.abs(self.curvature)) + 1e-8)
            ax.scatter(self.centerline[:, 0], self.centerline[:, 1],
                    c=c, cmap='cool', s=5, label='Curvature')
        else:
            ax.plot(self.centerline[:, 0], self.centerline[:, 1],
                    '-k', lw=2, label='Centerline')

        # Draw boundaries
        ax.plot(self.left_boundary[:, 0], self.left_boundary[:, 1],
                '--', color='gray', lw=1)
        ax.plot(self.right_boundary[:, 0], self.right_boundary[:, 1],
                '--', color='gray', lw=1)

        # Optional car trajectory overlay
        if car_traj is not None and len(car_traj) > 1:
            traj = np.array(car_traj)
            ax.plot(traj[:, 0], traj[:, 1], color='orange', lw=2, label='Trajectory')

        # Optional car state (position + heading)
        if car_state is not None:
            x, y, heading = car_state
            ax.plot(x, y, 'ro', markersize=8)
            ax.arrow(x, y,
                    2 * np.cos(heading), 2 * np.sin(heading),
                    head_width=0.5, color='r', length_includes_head=True,
                    alpha=0.8)

        ax.set_title(f"Track: {self.name}")
        ax.axis('equal')
        ax.legend(loc='best')
        plt.savefig(f"{self.name}_track.png", dpi=150)
        # If not embedded in another figure, save or show
        if ax is None:
            plt.tight_layout()
            plt.savefig(f"{self.name}_track.png", dpi=150)
            plt.close()



if __name__ == '__main__':
    # simple synthetic track
    theta = np.linspace(0, 2*np.pi, 500)
    waypoints = np.stack([50*np.cos(theta), 30*np.sin(theta)], axis=1)

    track = RaceTrack(waypoints, name="Oval", trackwidth=10)

    # test car position
    x, y, heading = 10, 26, 180

    track.visualize(show_curvature=True, car_state=(x,y,heading))
    print("Offtrack?", track.is_offtrack(x, y))
    s, d, h = track.nearest_point(x, y)
    print(f"s={s:.2f} m, lateral={d:.2f} m, heading={np.degrees(h):.1f}°")
