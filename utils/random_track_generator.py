# Standard library imports
import json
import math
import os
import random
import time
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from shapely import LineString


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distance2(p1, p2):
    """Calculate squared Euclidean distance between two points."""
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def sign(x):
    """Return the sign of a number."""
    return 1 if x >= 0 else -1


def get_centroid(points):
    """Calculate the centroid of a set of points."""
    cx = np.mean(points[0, :])
    cy = np.mean(points[1, :])
    return cx, cy


# =============================================================================
# TRACK VALIDATION FUNCTIONS
# =============================================================================

def no_intersections(points):
    """Check if points are spaced adequately without clustering."""
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


def no_hole_in_the_middle(points, radius):
    """Check if there's no large hole in the middle of the track."""
    cx, cy = get_centroid(points)
    for i in range(len(points)):
        if distance2([cx, cy], points[i]) < radius**2:
            return True
    return False


def check_spline_self_intersection(left_points, right_points):
    """Check if the generated track boundaries have self-intersections."""
    spline_left = LineString(left_points.T)
    spline_right = LineString(right_points.T)
    return spline_left.is_simple and spline_right.is_simple


# =============================================================================
# TRACK GENERATION FUNCTIONS
# =============================================================================

def random_walk():
    """Generate a random walk that forms a closed loop suitable for a race track."""
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
    """Fit a smooth spline to the given points."""
    tck, u = splprep(points, s=8.0, per=1, quiet=2)
    unew = np.linspace(0, 1, 1000)
    out = np.array(splev(unew, tck))
    return out, tck, unew


def get_turn_direction(p_prev, p_current, p_next):
    """Determine the turn direction at a point on the track."""
    v1 = p_current - p_prev
    v2 = p_next - p_current
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    if cross_product > 0:
        return -1  # left
    elif cross_product < 0:
        return 1  # right
    else:
        return 0  # straight


def generate_smooth_width_profile(num_points):
    """Generate a smooth varying width profile with mean 3.0 meters."""
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
    
    return width_profile


def generate_boundaries():
    """Generate track boundaries with smooth varying width."""
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

    # Generate smooth varying width profile
    num_points = np.size(track_points, axis=1)
    width_profile = generate_smooth_width_profile(num_points)
    width_shift = np.zeros(num_points)

    # Generate left and right boundaries' points
    left_points = np.empty_like(track_points)
    right_points = np.empty_like(track_points)
    
    for i in range(num_points):
        j = i + 1 if i + 1 < num_points else 0

        # Calculate perpendicular direction vector
        dir_vec = np.empty(2)
        dir_vec[0] = -(track_points[1][j] - track_points[1][i])
        dir_vec[1] = track_points[0][j] - track_points[0][i]
        dir_vec = dir_vec / np.linalg.norm(dir_vec)

        # Generate left boundary points
        sampling_lateral_distance = width_profile[i] / 2 + width_shift[i]
        left_points[0][i] = track_points[0][i] + sampling_lateral_distance * dir_vec[0]
        left_points[1][i] = track_points[1][i] + sampling_lateral_distance * dir_vec[1]
        
        # Generate right boundary points
        sampling_lateral_distance = width_profile[i] / 2 - width_shift[i]
        right_points[0][i] = track_points[0][i] - sampling_lateral_distance * dir_vec[0]
        right_points[1][i] = track_points[1][i] - sampling_lateral_distance * dir_vec[1]

    # Transform track to origin with proper orientation
    translation_vec = ((left_points[0][0] + right_points[0][0]) / 2.0, 
                      (left_points[1][0] + right_points[1][0]) / 2.0)
    left_points = np.array([left_points[0] - translation_vec[0], 
                           left_points[1] - translation_vec[1]])
    right_points = np.array([right_points[0] - translation_vec[0], 
                            right_points[1] - translation_vec[1]])
    
    # Rotate to align with x-axis
    forward_vec = ((left_points[0][1] + right_points[0][1]) / 2.0, 
                   (left_points[1][1] + right_points[1][1]) / 2.0)
    rot = -math.atan2(forward_vec[1], forward_vec[0])
    rotation_matrix = np.array([[math.cos(rot), -math.sin(rot)], 
                               [math.sin(rot), math.cos(rot)]])
    left_points = rotation_matrix.dot(left_points)
    right_points = rotation_matrix.dot(right_points)

    return left_points, right_points, track_points, curvature, width_profile


# =============================================================================
# EXPORT AND VISUALIZATION FUNCTIONS
# =============================================================================

def export_track(path, left_points, right_points):
    """Export track data to JSON format."""
    res = {"track": [], 'time_keeper': [[0.0, 0.0, 3.5]]}
    
    for i in range(len(left_points[0])):
        if i == 0:
            res["track"].append([left_points[0][0], left_points[1][0], 2])
            res["track"].append([right_points[0][0], right_points[1][0], 2])
            continue

        res["track"].append([left_points[0][i], left_points[1][i], 0])
        res["track"].append([right_points[0][i], right_points[1][i], 1])

    with open(path, "w") as f:
        json.dump(res, f)


def visualize_track(left_cones, right_cones, path):
    """Create and save a visualization of the generated track."""
    plt.figure(figsize=(12, 8))
    
    # Plot cones
    plt.scatter(left_cones[0, 0], left_cones[1, 0], c='orange', s=100, 
               label='Start/Finish Left', marker='s')
    plt.scatter(right_cones[0, 0], right_cones[1, 0], c='orange', s=100, 
               label='Start/Finish Right', marker='s')
    plt.scatter(left_cones[0, 1:], left_cones[1, 1:], c='blue', s=50, 
               label='Left Cones', alpha=0.7)
    plt.scatter(right_cones[0, 1:], right_cones[1, 1:], c='yellow', s=50, 
               label='Right Cones', alpha=0.7)
    
    # Draw track boundaries
    plt.plot(left_cones[0, :], left_cones[1, :], 'b-', alpha=0.5, linewidth=2)
    plt.plot(right_cones[0, :], right_cones[1, :], 'y-', alpha=0.5, linewidth=2)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Generated Random Track')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_track_view(track_view, track_id=None, view_id=None, save_path=None, show_plot=False):
    """Visualize a track view showing car pose, visible cones, and target path."""
    plt.figure(figsize=(12, 10))
    
    # Extract data from track view
    car_pose = track_view['car_pose']
    visible_left_cones = np.array(track_view['visible_left_cones'])
    visible_right_cones = np.array(track_view['visible_right_cones'])
    target_path = np.array(track_view['target_path'])
    
    # Plot visible cones
    if visible_left_cones.size > 0:
        plt.scatter(visible_left_cones[0], visible_left_cones[1], 
                   c='blue', s=80, label='Visible Left Cones', alpha=0.8, marker='o')
    
    if visible_right_cones.size > 0:
        plt.scatter(visible_right_cones[0], visible_right_cones[1], 
                   c='yellow', s=80, label='Visible Right Cones', alpha=0.8, marker='o')
    
    # Plot target path
    if target_path.size > 0:
        plt.plot(target_path[0], target_path[1], 'g-', linewidth=3, 
                label='Target Path', alpha=0.8)
        plt.scatter(target_path[0], target_path[1], 
                   c='green', s=30, alpha=0.6, marker='.')
    
    # Plot car position and orientation
    car_x, car_y = car_pose['x'], car_pose['y']
    car_heading = car_pose['heading']
    
    # Car position
    plt.scatter(car_x, car_y, c='red', s=200, marker='D', 
               label='Car Position', edgecolors='black', linewidth=2)
    
    # Car orientation arrow
    arrow_length = 2.0
    dx = arrow_length * np.cos(car_heading)
    dy = arrow_length * np.sin(car_heading)
    plt.arrow(car_x, car_y, dx, dy, head_width=0.5, head_length=0.3, 
             fc='red', ec='red', alpha=0.8, linewidth=2)
    
    # Add view range circles
    view_config = track_view.get('view_config', {})
    front_dist = view_config.get('view_distance_front', 25.0)
    back_dist = view_config.get('view_distance_back', 15.0)
    max_dist = max(front_dist, back_dist)
    
    # Plot view range circle
    circle = plt.Circle((car_x, car_y), max_dist, fill=False, 
                       color='gray', linestyle='--', alpha=0.5, linewidth=1)
    plt.gca().add_patch(circle)
    
    # Add directional view zones
    # Forward view zone
    forward_arc = plt.Circle((car_x, car_y), front_dist, fill=False, 
                            color='lightgreen', linestyle=':', alpha=0.7, linewidth=2)
    plt.gca().add_patch(forward_arc)
    
    # Backward view zone  
    backward_arc = plt.Circle((car_x, car_y), back_dist, fill=False, 
                             color='lightcoral', linestyle=':', alpha=0.7, linewidth=2)
    plt.gca().add_patch(backward_arc)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Title with metadata
    title = f'Track View'
    if track_id is not None:
        title += f' - Track {track_id}'
    if view_id is not None:
        title += f', View {view_id}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    
    # Add info text
    info_text = f'Car Heading: {np.degrees(car_heading):.1f}°\n'
    info_text += f'Visible Cones: L={len(visible_left_cones[0]) if visible_left_cones.size > 0 else 0}, '
    info_text += f'R={len(visible_right_cones[0]) if visible_right_cones.size > 0 else 0}\n'
    info_text += f'Path Points: {len(target_path[0]) if target_path.size > 0 else 0}\n'
    info_text += f'View Range: F={front_dist}m, B={back_dist}m'
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"View visualization saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_track_with_all_views(left_cones, right_cones, dataset_views, 
                                  track_id=None, save_path=None, show_plot=False):
    """Visualize the entire track with all sampled car positions and their view ranges."""
    plt.figure(figsize=(15, 12))
    
    # Plot the full track
    plt.scatter(left_cones[0, 0], left_cones[1, 0], c='orange', s=120, 
               label='Start/Finish Left', marker='s', edgecolors='black')
    plt.scatter(right_cones[0, 0], right_cones[1, 0], c='orange', s=120, 
               label='Start/Finish Right', marker='s', edgecolors='black')
    plt.scatter(left_cones[0, 1:], left_cones[1, 1:], c='blue', s=40, 
               label='Left Cones', alpha=0.7)
    plt.scatter(right_cones[0, 1:], right_cones[1, 1:], c='yellow', s=40, 
               label='Right Cones', alpha=0.7)
    
    # Draw track boundaries
    plt.plot(left_cones[0, :], left_cones[1, :], 'b-', alpha=0.4, linewidth=1.5)
    plt.plot(right_cones[0, :], right_cones[1, :], 'y-', alpha=0.4, linewidth=1.5)
    
    # Plot centerline
    centerline = compute_centerline(left_cones, right_cones)
    plt.plot(centerline[0, :], centerline[1, :], 'k--', alpha=0.3, linewidth=1, label='Centerline')
    
    # Plot all car positions and orientations
    car_positions_x = []
    car_positions_y = []
    
    for i, view in enumerate(dataset_views):
        car_pose = view['car_pose']
        car_x, car_y = car_pose['x'], car_pose['y']
        car_heading = car_pose['heading']
        
        car_positions_x.append(car_x)
        car_positions_y.append(car_y)
        
        # Plot car orientation with small arrows
        arrow_length = 1.0
        dx = arrow_length * np.cos(car_heading)
        dy = arrow_length * np.sin(car_heading)
        plt.arrow(car_x, car_y, dx, dy, head_width=0.3, head_length=0.2, 
                 fc='red', ec='red', alpha=0.6, linewidth=1)
        
        # Add view range circle (smaller for overview)
        view_config = view.get('view_config', {})
        max_dist = max(view_config.get('view_distance_front', 25.0),
                      view_config.get('view_distance_back', 15.0))
        circle = plt.Circle((car_x, car_y), max_dist, fill=False, 
                           color='red', alpha=0.1, linewidth=0.5)
        plt.gca().add_patch(circle)
    
    # Plot car positions
    plt.scatter(car_positions_x, car_positions_y, c='red', s=60, 
               label=f'Car Positions ({len(dataset_views)} views)', 
               alpha=0.7, marker='D', edgecolors='darkred')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    title = f'Track Overview with All Car Positions'
    if track_id is not None:
        title += f' - Track {track_id}'
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    
    # Add statistics
    stats_text = f'Total Views: {len(dataset_views)}\n'
    stats_text += f'Track Cones: L={left_cones.shape[1]}, R={right_cones.shape[1]}\n'
    stats_text += f'Car Position Spread: X=[{min(car_positions_x):.1f}, {max(car_positions_x):.1f}], '
    stats_text += f'Y=[{min(car_positions_y):.1f}, {max(car_positions_y):.1f}]'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Track overview saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_track(path):
    """Generate a complete track with boundaries and export it."""
    i = 0
    while True:
        i += 1
        left_points, right_points, track_points, curvature, width_profile = generate_boundaries()
        if check_spline_self_intersection(left_points, right_points):
            break

    # Generate cone positions along the boundaries
    left_tck, _ = splprep(left_points, s=0.0, quiet=2)
    right_tck, _ = splprep(right_points, s=0.0, quiet=2)
    left_cones = np.array(splev(np.linspace(0, 1, 100), left_tck))
    right_cones = np.array(splev(np.linspace(0, 1, 100), right_tck))
    
    # Remove duplicate last point
    left_cones = np.delete(left_cones, np.size(left_cones, axis=1) - 1, axis=1)
    right_cones = np.delete(right_cones, np.size(right_cones, axis=1) - 1, axis=1)

    # Visualize and export the track
    visualize_track(left_cones, right_cones, path)
    export_track(path, left_cones, right_cones)


# =============================================================================
# DATASET GENERATION FUNCTIONS
# =============================================================================

def compute_centerline(left_points, right_points):
    """Compute the centerline of the track from left and right boundaries."""
    centerline = np.zeros_like(left_points)
    centerline[0] = (left_points[0] + right_points[0]) / 2.0
    centerline[1] = (left_points[1] + right_points[1]) / 2.0
    return centerline


def get_tangent_at_point(centerline, idx):
    """Get the tangent vector at a specific point on the centerline."""
    num_points = centerline.shape[1]
    
    # Use neighboring points to compute tangent
    if idx == 0:
        tangent = centerline[:, 1] - centerline[:, -1]  # Use last point for circular track
    elif idx == num_points - 1:
        tangent = centerline[:, 0] - centerline[:, idx - 1]  # Use first point for circular track
    else:
        tangent = centerline[:, idx + 1] - centerline[:, idx - 1]
    
    # Normalize tangent vector
    tangent = tangent / np.linalg.norm(tangent)
    return tangent


def sample_car_pose(centerline, max_lateral_offset=1.0):
    """Sample a random car pose inside the track."""
    num_points = centerline.shape[1]
    
    # Sample random point along centerline
    centerline_idx = np.random.randint(0, num_points)
    centerline_point = centerline[:, centerline_idx]
    
    # Get tangent direction at this point
    tangent = get_tangent_at_point(centerline, centerline_idx)
    
    # Calculate perpendicular direction (normal)
    normal = np.array([-tangent[1], tangent[0]])
    
    # Sample lateral offset from centerline (with bias towards center)
    lateral_offset = np.random.uniform(-max_lateral_offset, max_lateral_offset)
    
    # Calculate car position
    car_position = centerline_point + lateral_offset * normal
    
    # Sample heading with small deviation from tangent direction
    tangent_angle = np.arctan2(tangent[1], tangent[0])
    heading_deviation = np.random.uniform(-np.pi/12, np.pi/12)  # ±15 degrees
    car_heading = tangent_angle + heading_deviation
    
    return {
        'position': car_position,
        'heading': car_heading,
        'centerline_idx': centerline_idx
    }


def get_visible_cones(car_pose, left_cones, right_cones, 
                     view_distance_front=25.0, view_distance_back=15.0):
    """Get cones visible from the car's current position and heading."""
    car_pos = car_pose['position']
    car_heading = car_pose['heading']
    
    # Direction vectors
    forward_vec = np.array([np.cos(car_heading), np.sin(car_heading)])
    
    visible_left_cones = []
    visible_right_cones = []
    
    # Check all left cones
    for i in range(left_cones.shape[1]):
        cone_pos = left_cones[:, i]
        relative_pos = cone_pos - car_pos
        distance = np.linalg.norm(relative_pos)
        
        # Check if cone is within view distance
        if distance <= max(view_distance_front, view_distance_back):
            # Check if cone is in front or behind
            forward_distance = np.dot(relative_pos, forward_vec)
            
            if (forward_distance >= 0 and forward_distance <= view_distance_front) or \
               (forward_distance < 0 and abs(forward_distance) <= view_distance_back):
                visible_left_cones.append(cone_pos)
    
    # Check all right cones
    for i in range(right_cones.shape[1]):
        cone_pos = right_cones[:, i]
        relative_pos = cone_pos - car_pos
        distance = np.linalg.norm(relative_pos)
        
        # Check if cone is within view distance
        if distance <= max(view_distance_front, view_distance_back):
            # Check if cone is in front or behind
            forward_distance = np.dot(relative_pos, forward_vec)
            
            if (forward_distance >= 0 and forward_distance <= view_distance_front) or \
               (forward_distance < 0 and abs(forward_distance) <= view_distance_back):
                visible_right_cones.append(cone_pos)
    
    return np.array(visible_left_cones).T if visible_left_cones else np.empty((2, 0)), \
           np.array(visible_right_cones).T if visible_right_cones else np.empty((2, 0))


def generate_target_path(car_pose, centerline, visible_left_cones, visible_right_cones):
    """Generate target path based on visible cones range."""
    car_pos = car_pose['position']
    car_heading = car_pose['heading']
    
    # Combine all visible cones to determine the range
    all_visible_cones = []
    if visible_left_cones.size > 0:
        all_visible_cones.extend(visible_left_cones.T)
    if visible_right_cones.size > 0:
        all_visible_cones.extend(visible_right_cones.T)
    
    if not all_visible_cones:
        # No visible cones, return minimal path around car position
        distances = np.linalg.norm(centerline - car_pos.reshape(-1, 1), axis=0)
        closest_idx = np.argmin(distances)
        return centerline[:, closest_idx:closest_idx+1]
    
    all_visible_cones = np.array(all_visible_cones)
    
    # Find centerline indices corresponding to the range of visible cones
    centerline_indices = []
    for cone in all_visible_cones:
        distances = np.linalg.norm(centerline - cone.reshape(-1, 1), axis=0)
        closest_idx = np.argmin(distances)
        centerline_indices.append(closest_idx)
    
    # Determine the range of indices to include in the path
    min_idx = min(centerline_indices)
    max_idx = max(centerline_indices)
    
    # Find car's closest position on centerline
    car_distances = np.linalg.norm(centerline - car_pos.reshape(-1, 1), axis=0)
    car_closest_idx = np.argmin(car_distances)
    
    # Determine direction of travel based on car heading
    forward_vec = np.array([np.cos(car_heading), np.sin(car_heading)])
    
    # Generate path points along centerline from the visible range
    # Handle circular track wraparound
    path_indices = []
    
    # Extend the range slightly beyond visible cones for smoother path
    range_extension = max(1, (max_idx - min_idx) // (1/0.03))  # Extend by 3% or at least 3 points
    
    if max_idx - min_idx < centerline.shape[1] // 2:
        # Normal case: visible range doesn't wrap around
        start_idx = max(0, min_idx - range_extension)
        end_idx = min(centerline.shape[1] - 1, max_idx + range_extension)
        path_indices = list(range(start_idx, end_idx + 1))
    else:
        # Handle wraparound case
        start_idx = (min_idx - range_extension) % centerline.shape[1]
        end_idx = (max_idx + range_extension) % centerline.shape[1]
        
        if start_idx <= end_idx:
            path_indices = list(range(start_idx, end_idx + 1))
        else:
            path_indices = list(range(start_idx, centerline.shape[1])) + list(range(0, end_idx + 1))
    
    # Create path points
    path_points = centerline[:, path_indices]
    
    return path_points


def create_track_view(car_pose, left_cones, right_cones, centerline, 
                     view_config=None):
    """Create a complete view of the track from the car's perspective."""
    if view_config is None:
        view_config = {
            'view_distance_front': 25.0,
            'view_distance_back': 15.0
        }
    
    # Get visible cones
    visible_left, visible_right = get_visible_cones(
        car_pose, left_cones, right_cones,
        view_config['view_distance_front'], 
        view_config['view_distance_back']
    )
    
    # Generate target path based on visible cones
    target_path = generate_target_path(
        car_pose, centerline, visible_left, visible_right
    )
    
    return {
        'car_pose': {
            'x': car_pose['position'][0],
            'y': car_pose['position'][1],
            'heading': car_pose['heading']
        },
        'visible_left_cones': visible_left.tolist() if visible_left.size > 0 else [],
        'visible_right_cones': visible_right.tolist() if visible_right.size > 0 else [],
        'target_path': target_path.tolist() if target_path.size > 0 else []
    }


def generate_dataset_from_track(left_cones, right_cones, num_views=50):
    """Generate multiple views from a single track for dataset creation."""
    centerline = compute_centerline(left_cones, right_cones)
    dataset_views = []
    
    # Define different view configurations for diversity
    view_configs = [
        {'view_distance_front': 25.0, 'view_distance_back': 15.0},
        {'view_distance_front': 30.0, 'view_distance_back': 10.0},
        {'view_distance_front': 15.0, 'view_distance_back': 5.0},
        {'view_distance_front': 35.0, 'view_distance_back': 20.0},
        {'view_distance_front': 5.0, 'view_distance_back': 5.0}  # Edge case
    ]
    
    for i in range(num_views):
        # Sample random car pose
        car_pose = sample_car_pose(centerline)
        
        # Select random view configuration for diversity
        view_config = view_configs[i % len(view_configs)]
        
        # Create track view
        track_view = create_track_view(car_pose, left_cones, right_cones, centerline, view_config)
        
        # Add metadata
        track_view['view_id'] = i
        track_view['view_config'] = view_config
        
        dataset_views.append(track_view)
    
    return dataset_views


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def generate_dataset(num_tracks=10, views_per_track=50, dataset_path="./dataset", 
                    visualize_views=False, save_view_plots=False, views_to_plot=5):
    """Generate a complete dataset with multiple tracks and views for training."""
    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(f"{dataset_path}/tracks", exist_ok=True)
    os.makedirs(f"{dataset_path}/views", exist_ok=True)
    
    if save_view_plots:
        os.makedirs(f"{dataset_path}/view_plots", exist_ok=True)
        os.makedirs(f"{dataset_path}/track_overviews", exist_ok=True)
    
    all_dataset_views = []
    
    print(f"Generating dataset with {num_tracks} tracks and {views_per_track} views per track...")
    if save_view_plots:
        print(f"  Will save visualizations for first {views_to_plot} views per track")
    start_time = time.time()
    
    for track_id in range(num_tracks):
        print(f"Generating track {track_id + 1}/{num_tracks}...")
        
        # Generate track boundaries until we get a valid one
        i = 0
        while True:
            i += 1
            left_points, right_points, track_points, curvature, width_profile = generate_boundaries()
            if check_spline_self_intersection(left_points, right_points):
                break
        
        # Generate cone positions along the boundaries
        left_tck, _ = splprep(left_points, s=0.0, quiet=2)
        right_tck, _ = splprep(right_points, s=0.0, quiet=2)
        left_cones = np.array(splev(np.linspace(0, 1, 100), left_tck))
        right_cones = np.array(splev(np.linspace(0, 1, 100), right_tck))
        
        # Remove duplicate last point
        left_cones = np.delete(left_cones, np.size(left_cones, axis=1) - 1, axis=1)
        right_cones = np.delete(right_cones, np.size(right_cones, axis=1) - 1, axis=1)
        
        # Save track visualization
        track_path = f"{dataset_path}/tracks/track_{track_id:03d}"
        visualize_track(left_cones, right_cones, track_path)
        export_track(track_path, left_cones, right_cones)
        
        # Generate views from this track
        print(f"  Generating {views_per_track} views from track {track_id + 1}...")
        track_views = generate_dataset_from_track(left_cones, right_cones, views_per_track)
        
        # Add track metadata to each view
        for view in track_views:
            view['track_id'] = track_id
            view['global_view_id'] = len(all_dataset_views)
            all_dataset_views.append(view)
        
        # Save views for this track
        track_views_path = f"{dataset_path}/views/track_{track_id:03d}_views.json"
        with open(track_views_path, 'w') as f:
            json.dump(track_views, f, indent=2)
        
        # Generate visualizations if requested
        if save_view_plots:
            # Save overview of track with all car positions
            overview_path = f"{dataset_path}/track_overviews/track_{track_id:03d}_overview.png"
            visualize_track_with_all_views(left_cones, right_cones, track_views, 
                                         track_id=track_id, save_path=overview_path)
            
            # Save individual view visualizations for first few views
            print(f"  Saving {min(views_to_plot, len(track_views))} view visualizations...")
            for i in range(min(views_to_plot, len(track_views))):
                view_plot_path = f"{dataset_path}/view_plots/track_{track_id:03d}_view_{i:03d}.png"
                visualize_track_view(track_views[i], track_id=track_id, view_id=i, 
                                   save_path=view_plot_path)
    
    # Save complete dataset
    dataset_file = f"{dataset_path}/complete_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(all_dataset_views, f, indent=2)
    
    # Save dataset metadata
    metadata = {
        'num_tracks': num_tracks,
        'views_per_track': views_per_track,
        'total_views': len(all_dataset_views),
        'generation_time': time.time() - start_time,
        'dataset_structure': {
            'car_pose': 'Car position (x, y) and heading angle in radians',
            'visible_left_cones': 'List of [x, y] coordinates of visible left cones',
            'visible_right_cones': 'List of [x, y] coordinates of visible right cones', 
            'target_path': 'List of [x, y] coordinates of target path points (automatically spans from first to last visible cone)',
            'view_config': 'Configuration used for this view (front/back view distances)'
        }
    }
    
    metadata_file = f"{dataset_path}/dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"Total tracks generated: {num_tracks}")
    print(f"Total views generated: {len(all_dataset_views)}")
    print(f"Total generation time: {total_time:.2f} seconds")
    print(f"Average time per track: {total_time / num_tracks:.2f} seconds")
    print(f"Average time per view: {total_time / len(all_dataset_views):.4f} seconds")
    print(f"Dataset saved to: {dataset_path}/")
    print(f"Main dataset file: {dataset_file}")
    print(f"Dataset metadata: {metadata_file}")


def main():
    """Generate multiple random tracks or a complete dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate random race tracks and datasets')
    parser.add_argument('--mode', choices=['tracks', 'dataset'], default='tracks',
                       help='Generation mode: tracks (visualizations only) or dataset (for ML training)')
    parser.add_argument('--num-tracks', type=int, default=5,
                       help='Number of tracks to generate')
    parser.add_argument('--views-per-track', type=int, default=50,
                       help='Number of views per track (dataset mode only)')
    parser.add_argument('--output', type=str, default='./plots',
                       help='Output directory')
    
    # Visualization arguments for dataset mode
    parser.add_argument('--save-view-plots', action='store_true',
                       help='Save individual view plots and track overviews (dataset mode only)')
    parser.add_argument('--views-to-plot', type=int, default=5,
                       help='Number of views per track to save as plots (requires --save-view-plots)')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots interactively (not recommended for large datasets)')
    
    args = parser.parse_args()
    
    if args.mode == 'dataset':
        # Generate dataset for ML training
        dataset_path = args.output if args.output != './plots' else './dataset'
        generate_dataset(
            num_tracks=args.num_tracks, 
            views_per_track=args.views_per_track, 
            dataset_path=dataset_path,
            visualize_views=args.show_plots,
            save_view_plots=args.save_view_plots,
            views_to_plot=args.views_to_plot
        )
    else:
        # Generate tracks for visualization only
        base_destination = f"{args.output}/random_track"
        os.makedirs(os.path.dirname(base_destination), exist_ok=True)
        
        print(f"Generating {args.num_tracks} random tracks...")
        start_time = time.time()
        
        for i in range(args.num_tracks):
            destination = f"{base_destination}_{i+1:02d}"
            print(f"Generating track {i+1}/{args.num_tracks}...")
            generate_track(destination)
            
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time per track: {total_time / args.num_tracks:.2f} seconds")
        print(f"Successfully generated {args.num_tracks} tracks!")
        print(f"Files saved in: {os.path.dirname(base_destination)}/")

if __name__ == "__main__":
    main()