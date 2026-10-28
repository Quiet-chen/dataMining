#!/usr/bin/env bash
# Download the authors' checkpoints from a fast S3 endpoint.
#
# get_checkpoint.py uses the regional endpoint scifact.s3.us-west-2.amazonaws.com,
# which resolves to IPs that are currently ~40 KB/s from this machine.  The
# global endpoint resolves to several IPs; 3.5.87.79 gave ~10 MB/s.  We pin that
# IP with curl --resolve (TLS SNI and the Host header stay scifact.s3.amazonaws.com)
# and resume with -C - if the connection drops.
set -u

HOST="scifact.s3.amazonaws.com"
IP="3.5.87.79"
BASE="https://${HOST}/longchecker/latest/checkpoints"
FILES=(healthver.ckpt longformer_large_science.ckpt)

declare -A SIZES=(
  [healthver.ckpt]=5253768926
  [longformer_large_science.ckpt]=1646843693
)

cd "$(dirname "$0")"

for name in "${FILES[@]}"; do
  want="${SIZES[$name]}"
  while true; do
    have=$(stat -c %s "$name" 2>/dev/null || echo 0)
    if [ "$have" -eq "$want" ]; then
      echo "$(date '+%F %T') DONE $name ($have bytes)"
      break
    fi
    echo "$(date '+%F %T') GET $name from $have/$want"
    # Abort and retry if the pinned IP becomes slow (<300 KB/s for 60 s).
    curl -sS --resolve "${HOST}:443:${IP}" -C - -o "$name" \
      --speed-time 60 --speed-limit 300000 "${BASE}/${name}" || true
    sleep 3
  done
done

echo "$(date '+%F %T') ALL DONE"
