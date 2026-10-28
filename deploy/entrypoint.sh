#!/bin/sh
set -eu

cd /app

if [ -f /app/config/settings.yaml ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    key=${line%%:*}
    value=${line#*:}
    key=$(printf '%s' "$key" | tr -d '[:space:]')
    value=$(printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')
    if [ -z "$key" ] || [ -z "$value" ]; then
      continue
    fi
    eval "current=\${$key-}"
    if [ -z "$current" ]; then
      export "$key=$value"
    fi
  done < /app/config/settings.yaml
fi

for key in \
  APP_SCRIPT CAMERAS_CONFIG CAMERAS_JSON CAMERAS \
  CAMERA_ID CAMERA_URL CAMERA_RTSP \
  CAMERA_202_URL CAMERA_202_RTSP CAMERA_302_URL CAMERA_302_RTSP \
  CAMERA_401_URL CAMERA_401_RTSP \
  STAFF_GALLERY_PATH V4_YOLO_MODEL V4_YOLO_IMGSZ V4_YOLO_CONF \
  V3_FACE_INTERVAL_SEC V4_GLOBAL_TTL_SEC V5_LOCAL_TRACK_REUSE_SEC \
  V4_BODY_REID V4_REID_CHECKPOINT V4_FACE_VOTES_REQUIRED \
  V4_FACE_VOTE_WINDOW_SEC V4_BODY_MATCH_THRESHOLD \
  V4_FACE_MATCH_THRESHOLD V4_FACE_MARGIN
do
  eval "current=\${$key-}"
  if [ -z "$current" ]; then
    unset "$key" || true
  fi
done

script=${APP_SCRIPT:-scripts/multi_camera_staff_tracking_v5_safe.py}
exec xvfb-run -a python "$script"
