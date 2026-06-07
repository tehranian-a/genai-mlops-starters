#!/usr/bin/env bash
set -euo pipefail

# Example only.
# Usage:
# ./registry_mirror.sh registry.example.local/genai image-list.txt

TARGET_REGISTRY="${1:-registry.example.local/genai}"
IMAGE_LIST="${2:-image-list.txt}"

while read -r IMAGE; do
  if [[ -z "$IMAGE" || "$IMAGE" == \#* ]]; then
    continue
  fi

  IMAGE_NAME="$(basename "$IMAGE")"
  TARGET_IMAGE="$TARGET_REGISTRY/$IMAGE_NAME"

  echo "Pulling: $IMAGE"
  docker pull "$IMAGE"

  echo "Tagging: $IMAGE -> $TARGET_IMAGE"
  docker tag "$IMAGE" "$TARGET_IMAGE"

  echo "Pushing: $TARGET_IMAGE"
  docker push "$TARGET_IMAGE"

done < "$IMAGE_LIST"
