#!/bin/bash

IMAGE_NAME="panjd123/vllama"

VERSIONS=(
  v0.8.5.post1  # cuda12.4
  v0.11.0 # cuda12.8
  latest  # cuda12.9
)
# ===========================


set -e

for VERSION in "${VERSIONS[@]}"; do
    echo "==============================="
    echo "🚀 Building $IMAGE_NAME:$VERSION"
    echo "==============================="

    docker build \
        --build-arg BASE_TAG="$VERSION" \
        --network host \
        -t "$IMAGE_NAME:$VERSION" .

    echo "📤 Pushing $IMAGE_NAME:$VERSION"
    docker push "$IMAGE_NAME:$VERSION"
done

echo "🎉 All images built & pushed successfully!"