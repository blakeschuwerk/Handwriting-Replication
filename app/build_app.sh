#!/bin/bash
# Builds "Handwriting cloner.app" into /Applications from launcher.applescript.
# Re-run after editing the script or moving the project.
set -e
cd "$(dirname "$0")"
APP="/Applications/Handwriting cloner.app"
rm -rf "$APP"
osacompile -s -o "$APP" launcher.applescript

# icon: ink-blue rounded square with a cursive "Hc"
TMP=$(mktemp -d)
../.venv/bin/python3 - "$TMP" <<'PY'
import sys
from PIL import Image, ImageDraw, ImageFont
S = 1024
bg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
g = Image.new("RGB", (S, S))
px = g.load()
for y in range(S):
    for x in range(S):
        t = (x + y) / (2 * S)
        px[x, y] = (int(34 + 20 * t), int(40 + 24 * t), int(78 + 52 * t))
m = Image.new("L", (S, S), 0)
ImageDraw.Draw(m).rounded_rectangle((56, 56, S - 56, S - 56), radius=225, fill=255)
bg.paste(g, (0, 0), m)
d = ImageDraw.Draw(bg)
f = ImageFont.truetype("/System/Library/Fonts/Supplemental/SnellRoundhand.ttc", 640, index=1)
w = d.textlength("Hc", font=f)
d.text(((S - w) / 2, 150), "Hc", font=f, fill=(244, 240, 228))
d.line((250, 790, 774, 790), fill=(244, 240, 228, 180), width=6)
bg.save(sys.argv[1] + "/icon.png")
PY
mkdir "$TMP/i.iconset"
for s in 16 32 128 256 512; do
  sips -z $s $s "$TMP/icon.png" --out "$TMP/i.iconset/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) "$TMP/icon.png" --out "$TMP/i.iconset/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$TMP/i.iconset" -o "$APP/Contents/Resources/applet.icns"
rm -rf "$TMP"
codesign --force --deep -s - "$APP"
touch "$APP"
echo "built $APP"
