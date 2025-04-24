#!/usr/bin/env bash
export DISPLAY=:0

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Opening Google..."
  # Use full path if explorer.exe isn’t in PATH:
  explorer.exe "https://www.google.com" &         # Launches default browser :contentReference[oaicite:4]{index=4}
  
  sleep 30                                         # Give time for page to load / generate traffic :contentReference[oaicite:5]{index=5}
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Closing Google..."
  powershell.exe -NoProfile -NonInteractive -Command \
    "Get-Process msedge -ErrorAction SilentlyContinue | Stop-Process -Force"
  
  sleep 270                                        # Finish the 5-minute interval
done