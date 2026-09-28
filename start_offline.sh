#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
image="${FALCON_IMAGE:-falcon-chel:final-20260928}"
if ! docker image inspect "$image" >/dev/null 2>&1; then
    archive="${1:-../falcon-chel-docker-20260928.tar.gz}"
    if [ ! -f "$archive" ]; then
        echo "Missing local image $image and archive $archive" >&2
        echo "Run: sh start_offline.sh /path/to/falcon-chel-docker-20260928.tar.gz" >&2
        exit 1
    fi
    docker load --input "$archive"
fi
docker image inspect "$image" >/dev/null
docker compose config --quiet
docker compose up --detach --no-build --pull never --wait --wait-timeout 240
echo "Falcon: http://127.0.0.1:${FALCON_PORT:-8799}"
echo "Swagger UI: http://127.0.0.1:${FALCON_PORT:-8799}/docs"
