#!/bin/bash
# Download OpenNeuro ds008701 (MET) from the public S3 bucket. Usage: bash download_met.sh <dest_dir>
set -e
DEST=${1:-$HOME/music_therapy/data/ds008701}; mkdir -p "$DEST"; cd "$DEST"
B=https://s3.amazonaws.com/openneuro.org
python3 - <<'PY' > /tmp/met_keys.txt
import urllib.request, urllib.parse, re
keys=[]; marker=""
while True:
    u="https://s3.amazonaws.com/openneuro.org/?list-type=2&prefix=ds008701/&max-keys=1000"+(("&continuation-token="+urllib.parse.quote(marker,safe="")) if marker else "")
    x=urllib.request.urlopen(u).read().decode()
    keys+=re.findall(r"<Key>([^<]+)</Key>",x)
    m=re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>",x)
    if not m: break
    marker=m.group(1)
print("\n".join(keys))
PY
wc -l /tmp/met_keys.txt
cat /tmp/met_keys.txt | xargs -P 256 -I{} sh -c 'f="{}"; f=${f#ds008701/}; mkdir -p "$(dirname "$f")"; [ -s "$f" ] || { curl -sfL --retry 5 "'$B'/ds008701/$f" -o "$f.part" && mv "$f.part" "$f"; }'
du -sh .; find . -name '*.set' | wc -l
