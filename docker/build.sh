#!/bin/bash
set -e

# Configuration
IMAGE_NAME="meshwell-mmg"
DOCKER_DIR="$(dirname "$0")"
OUTPUT_DIR="${1:-./bin}"  # Default output directory is ./bin in current dir

echo "Building Docker image: $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" "$DOCKER_DIR"

echo "Build complete."

if [ -n "$OUTPUT_DIR" ]; then
    echo "Extracting binaries to $OUTPUT_DIR..."
    mkdir -p "$OUTPUT_DIR"
    
    # Create a temporary container
    CONTAINER_ID=$(docker create "$IMAGE_NAME")
    
    # Copy binaries
    docker cp "$CONTAINER_ID":/usr/local/bin/mmg2d_O3 "$OUTPUT_DIR/"
    docker cp "$CONTAINER_ID":/usr/local/bin/mmg3d_O3 "$OUTPUT_DIR/"
    docker cp "$CONTAINER_ID":/usr/local/bin/mmgs_O3 "$OUTPUT_DIR/"
    docker cp "$CONTAINER_ID":/usr/local/bin/parmmg_O3 "$OUTPUT_DIR/"
    
    # Clean up
    docker rm "$CONTAINER_ID"
    
    echo "Binaries extracted successfully:"
    ls -l "$OUTPUT_DIR"
fi
