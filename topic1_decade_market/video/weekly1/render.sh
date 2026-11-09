#!/bin/bash
# renders frames/NN.html to frames/NN.png (1920x1080) with headless Chrome
cd "$(dirname "$0")"; for f in frames/*.html; do "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --window-size=1920,1080 --screenshot="${f%.html}.png" "file://$PWD/$f" >/dev/null 2>&1; done
