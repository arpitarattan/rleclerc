from circuit2waypoint import extract_track_from_image
from track import RaceTrack
from simplecar import SimpleCar
import numpy as np
import matplotlib.pyplot as plt
from env import RacingEnv

# ============= Test track
waypoints = extract_track_from_image(
    "tracks/silverstone.jpeg",
    invert=True,
    plot_steps=False
)
print('Extracted waypoints')
track = RaceTrack(waypoints, name="Silverstone", trackwidth=15)
print(np.min(track.curvature), np.max(track.curvature), np.mean(track.curvature))
track.visualize(show_curvature=True)

# ============== Test Environment
# STEPS = 500
# theta = np.linspace(0, 2*np.pi, 2000)
# waypoints = np.stack([50*np.cos(theta), 30*np.sin(theta)], axis=1)

# track = RaceTrack(waypoints, trackwidth=10)
# car = SimpleCar(init_pos=track.centerline[0])
# env = RacingEnv(track, car, max_steps = STEPS)

# obs = env.reset()
# rewards = []
# for step in range(STEPS):
#     action = [0.2, 0.3]  # constant throttle, mild turn
#     obs, reward, done, info = env.step(action)
#     rewards.append(reward)
#     if step % 3 == 0:
#         env.render(mode="gif")
#     if done:
#         break

# env.save_gif("race.gif", fps=20)

# plt.plot(rewards)
# plt.savefig('rewards.jpg')
