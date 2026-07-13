#!/bin/bash

PROJECT_ROOT="/data/projects/udl-raayon"
ROS2_WS="$PROJECT_ROOT/ros2_ws"
VENV="$PROJECT_ROOT/venv"

# Parse arguments
BUILD=false
for arg in "$@"; do
    case $arg in
        --build) BUILD=true ;;
    esac
done

# Activate venv
source "$VENV/bin/activate"

# Export venv packages to PYTHONPATH
export PYTHONPATH="$VENV/lib/python3.10/site-packages:$PYTHONPATH"

# Source ROS 2
source /opt/ros/humble/setup.bash

if [ "$BUILD" = true ]; then
    echo "Cleaning build artifacts..."
    rm -rf "$ROS2_WS/build" "$ROS2_WS/install" "$ROS2_WS/log"

    echo "Building detection_pkg..."
    cd "$ROS2_WS"
    colcon build --packages-select detection_pkg

    if [ $? -ne 0 ]; then
        echo "Build failed. Exiting."
        exit 1
    fi
fi

# Source workspace
source "$ROS2_WS/install/setup.bash"

echo "Launching person detector..."
# ros2 run detection_pkg person_detector --ros-args -p camera_topic:=/camera -p confidence_threshold:=0.35
# ros2 run detection_pkg green_detector --ros-args -p camera_topic:=/camera
ros2 run detection_pkg aruco_detector
