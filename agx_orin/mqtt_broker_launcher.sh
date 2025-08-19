#!/usr/bin/env bash
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Run with sudo or as root"
  exit 1
fi

CONF="/etc/mosquitto/conf.d/remote.conf"

if ! command -v mosquitto >/dev/null 2>&1; then
  echo "[INFO] mosquitto not found; installing..."
  apt update
  DEBIAN_FRONTEND=noninteractive apt install -y mosquitto
fi

cat <<'EOF' > "$CONF"
listener 1883 0.0.0.0
allow_anonymous true
EOF

echo "[INFO] Wrote config to $CONF"

systemctl enable --now mosquitto
systemctl restart mosquitto

echo "[INFO] mosquitto service status:"
systemctl status mosquitto --no-pager

echo "[INFO] Listening sockets on 1883:"
ss -ltnp | grep 1883 || true