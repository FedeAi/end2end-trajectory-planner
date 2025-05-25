import os, sys, json, math, random
import shutil
import time
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev, splrep
from scipy.ndimage import gaussian_filter1d
from shapely import LineString


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distance2(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def no_intersections(points):
    for i in range(len(points)):
        intersect = 0
        for j in range(len(points)):
            if i == j:
                continue
            if distance2(points[i], points[j]) < 2.56:  # 1.6**2
                intersect += 1
        if intersect > 2:
            return False
    return True


def get_centroid(points):
    cx = np.mean(points[0, :])
    cy = np.mean(points[1, :])
    return cx, cy


def no_hole_in_the_middle(points, radius):
    cx, cy = get_centroid(points)
    for i in range(len(points)):
        if distance2([cx, cy], points[i]) < radius**2:
            return True
    return False


def sign(x):
    return 1 if x >= 0 else -1


def random_walk():
    steps = 200
    step_length = 20.0
    delta_angle = math.pi / 6

    attempts = 1
    while True:
        points = np.zeros((steps, 2))
        pose_x, pose_y = 2000, 2000
        current_angle = 0.0
        points[0] = [pose_x / 10.0, pose_y / 10.0]
        for i in range(1, steps):
            if i <= 150:
                center_angle = 0.0
            else:
                a = math.atan2(points[i - 1][1], points[i - 1][0])
                b = math.atan2(points[i - 1][1] - points[i - 2][1], points[i - 1][0] - points[i - 2][0])
                center_angle = (a - b) * 0.1
            current_angle += random.uniform(center_angle - delta_angle, center_angle + delta_angle)
            pose_x += int(step_length * math.cos(current_angle))
            pose_y += int(step_length * math.sin(current_angle))
            if pose_x < 0 or pose_x >= 4000 or pose_y < 0 or pose_y >= 4000:
                break
            points[i] = [pose_x / 10.0, pose_y / 10.0]
            if i > 150 and distance2(points[0], points[i]) < 9:
                points = points[:i]
                return points.T
        attempts += 1


def fit_spline(points):
    tck, u = splprep(points, s=8.0, per=1, quiet=2)
    unew = np.linspace(0, 1, 1000)
    out = np.array(splev(unew, tck))
    return out, tck, unew


def check_spline_self_intersection(left_points, right_points):
    spline_left = LineString(left_points.T)
    spline_right = LineString(right_points.T)
    return spline_left.is_simple and spline_right.is_simple


def get_turn_direction(p_prev, p_current, p_next):
    v1 = p_current - p_prev
    v2 = p_next - p_current
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    if cross_product > 0:
        return -1  # left
    elif cross_product < 0:
        return 1  # right
    else:
        return 0  # straight


def generate_boundaries():
    points = random_walk()
    track_points, tck, u = fit_spline(points)

    # Compute curvature of the spline
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

    # Shift starting point to straight section
    curvature_threshold = np.percentile(np.abs(curvature), 0.15)
    candidate_start_idxs = np.where(np.abs(curvature) < curvature_threshold)
    start_idx = np.random.choice(candidate_start_idxs[0])
    track_points = np.delete(track_points, np.size(track_points, axis=1) - 1, axis=1)
    track_points = np.roll(track_points, -start_idx, axis=1)

    # Generate smooth varying width profile with mean 3.0
    num_points = np.size(track_points, axis=1)
    
    # Create smooth variations using multiple sine waves with different frequencies
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    
    # Combine multiple frequency components for natural variation
    width_variation = (0.4 * np.sin(1.5 * t) +           # Low frequency variation
                      0.3 * np.sin(3.7 * t) +            # Medium frequency variation  
                      0.1 * np.sin(7.2 * t))             # High frequency variation
    
    # Apply smoothing to ensure very smooth transitions
    width_variation = gaussian_filter1d(width_variation, sigma=2.0, mode='wrap')
    
    # Create width profile: base width of 3.0 + smooth variations
    # Width will vary between approximately 2.2 and 3.8 meters
    width_profile = 3.0 + width_variation
    
    # Ensure minimum width of 2.0 meters for safety
    width_profile = np.maximum(width_profile, 2.0)
    
    width_shift = np.zeros(np.size(track_points, axis=1))

    # Generate left and right boundaries' points
    left_points = np.empty_like(track_points)
    right_points = np.empty_like(track_points)
    for i in range(np.size(track_points, axis=1)):
        j = i + 1 if i + 1 < np.size(track_points, axis=1) else 0

        dir_vec = np.empty(2)
        dir_vec[0] = -(track_points[1][j] - track_points[1][i])
        dir_vec[1] = track_points[0][j] - track_points[0][i]
        dir_vec = dir_vec/np.linalg.norm(dir_vec)

        sampling_lateral_distance = width_profile[i] / 2 + width_shift[i]
        left_points[0][i] = track_points[0][i] + sampling_lateral_distance * dir_vec[0]
        left_points[1][i] = track_points[1][i] + sampling_lateral_distance * dir_vec[1]
        sampling_lateral_distance = width_profile[i] / 2 - width_shift[i]
        right_points[0][i] = track_points[0][i] - sampling_lateral_distance * dir_vec[0]
        right_points[1][i] = track_points[1][i] - sampling_lateral_distance * dir_vec[1]

    # Translate all points to the origin
    translation_vec = ((left_points[0][0] + right_points[0][0]) / 2.0, (left_points[1][0] + right_points[1][0]) / 2.0)
    left_points = np.array([left_points[0] - translation_vec[0], left_points[1] - translation_vec[1]])
    right_points = np.array([right_points[0] - translation_vec[0], right_points[1] - translation_vec[1]])
    forward_vec = ((left_points[0][0 + 1] + right_points[0][0 + 1]) / 2.0, (left_points[1][0 + 1] + right_points[1][0 + 1]) / 2.0)
    rot = -math.atan2(forward_vec[1], forward_vec[0])
    left_points = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]).dot(left_points)
    right_points = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]).dot(right_points)

    return left_points, right_points, track_points, curvature


def export_track(path, left_points, right_points):
    res = {"track": [], 'time_keeper': [[0.0, 0.0, 3.5]]}
    for i in range(len(left_points[0])):
        if i == 0:
            res["track"].append([left_points[0][0], left_points[1][0], 2])
            res["track"].append([right_points[0][0], right_points[1][0], 2])
            continue

        res["track"].append([left_points[0][i], left_points[1][i], 0])
        res["track"].append([right_points[0][i], right_points[1][i], 1])

    j = json.dumps(res)
    with open(path, "w") as f:
        f.write(j)


def generate_track(path):
    while True:
        left_points, right_points, track_points, curvature = generate_boundaries()
        if check_spline_self_intersection(left_points, right_points):
            break

    left_tck, _ = splprep(left_points, s=0.0, quiet=2)
    right_tck, _ = splprep(right_points, s=0.0, quiet=2)
    left_cones = np.array(splev(np.linspace(0, 1, 100), left_tck))
    right_cones = np.array(splev(np.linspace(0, 1, 100), right_tck))
    left_cones = np.delete(left_cones, np.size(left_cones, axis=1) - 1, axis=1)
    right_cones = np.delete(right_cones, np.size(right_cones, axis=1) - 1, axis=1)

    # Visualize the track
    plt.figure(figsize=(12, 8))
    plt.scatter(left_cones[0, 0], left_cones[1, 0], c='orange', s=100, label='Start/Finish Left', marker='s')
    plt.scatter(right_cones[0, 0], right_cones[1, 0], c='orange', s=100, label='Start/Finish Right', marker='s')
    plt.scatter(left_cones[0, 1:], left_cones[1, 1:], c='blue', s=50, label='Left Cones', alpha=0.7)
    plt.scatter(right_cones[0, 1:], right_cones[1, 1:], c='yellow', s=50, label='Right Cones', alpha=0.7)
    
    # Draw the track boundaries as lines
    plt.plot(left_cones[0, :], left_cones[1, :], 'b-', alpha=0.5, linewidth=2)
    plt.plot(right_cones[0, :], right_cones[1, :], 'y-', alpha=0.5, linewidth=2)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Generated Random Track')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.savefig(f'{path}.png', dpi=300)
    plt.close()

    export_track(path, left_cones, right_cones)


def main():
    # Generate 30 different tracks
    num_tracks = 5
    base_destination = "./plots/random_track"
    os.makedirs(os.path.dirname(base_destination), exist_ok=True)
    
    print(f"Generating {num_tracks} random tracks...")
    start_time = time.time()
    for i in range(num_tracks):
        destination = f"{base_destination}_{i+1:02d}"
        print(f"Generating track {i+1}/{num_tracks}...")
        generate_track(destination)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time  per track: {total_time / num_tracks:.2f} seconds")
    print(f"Successfully generated {num_tracks} tracks!")
    print(f"Files saved in: {os.path.dirname(base_destination)}/")


if __name__ == '__main__':
    main()