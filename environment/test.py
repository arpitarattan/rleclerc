from circuit2waypoint import extract_track_from_image
from track import RaceTrack
import numpy as np
waypoints = extract_track_from_image(
    "silverstone.jpeg",
    invert=True,
    plot_steps=False
)
print('Extracted waypoints')
track = RaceTrack(waypoints, name="Silverstone", trackwidth=10)
print(np.min(track.curvature), np.max(track.curvature), np.mean(track.curvature))


track.visualize(show_curvature=True)
