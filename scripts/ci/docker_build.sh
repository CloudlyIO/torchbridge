#!/bin/bash
# Build a TorchBridge Docker image with the version from pyproject.toml.
#
# Usage:
#   scripts/ci/docker_build.sh docker/Dockerfile.nvidia torchbridge:nvidia
#
# The script extracts the version from pyproject.toml and passes it as
# TORCHBRIDGE_VERSION build arg so Docker LABELs stay in sync automatically.

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <Dockerfile> <tag> [extra docker build args...]"
    exit 1
fi

DOCKERFILE="$1"
TAG="$2"
shift 2

VERSION=$(python3 -c "import re; print(re.search(r'^version\s*=\s*\"([^\"]+)\"', open('pyproject.toml').read(), re.MULTILINE).group(1))")

echo "Building ${TAG} (v${VERSION}) from ${DOCKERFILE}"

docker build \
    --build-arg TORCHBRIDGE_VERSION="${VERSION}" \
    -f "${DOCKERFILE}" \
    -t "${TAG}" \
    "$@" \
    .
