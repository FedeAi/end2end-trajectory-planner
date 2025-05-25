# End-to-End Trajectory Planner

A Python-based random track generator for autonomous vehicle trajectory planning and simulation.

## Features

- Generates random closed-loop racing tracks using spline interpolation
- Creates smooth, collision-free track boundaries
- Visualizes tracks with matplotlib
- Exports tracks in custom SYTD format for simulation

## Requirements

- Python 3.7+
- Virtual environment (recommended)

## Installation

### 1. Clone or Navigate to Project Directory

```bash
cd /home/claudio/Documents/fede/end2end-trajectory-planner
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Generate a Random Track

With the virtual environment activated:

```bash
python utils/random_track_generator.py
```

This will:
1. Generate a random closed-loop track
2. Display a visualization of the track with matplotlib
3. Save the track data to a `.sytd` file and PNG image

### Track Visualization

The generated track includes:
- **Orange squares**: Start/finish line markers
- **Blue dots**: Left boundary cones
- **Yellow dots**: Right boundary cones
- **Connecting lines**: Track boundaries for better visibility

## Dependencies

- **matplotlib**: Track visualization and plotting
- **numpy**: Numerical computations and array operations
- **scipy**: Spline interpolation functions
- **shapely**: Geometric operations and intersection checking

