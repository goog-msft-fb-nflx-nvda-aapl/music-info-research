#!/bin/bash
# Download an OpenNeuro dataset from the public S3 bucket. Usage: bash download_met.sh <dest_dir> [dataset_id=ds008701]
set -e
DS=${2:-ds008701}
DEST=${1:-$HOME/music_therapy/data/$DS}; mkdir -p "$DEST"; cd "$DEST"
B=https://s3.amazonaws.com/openneuro.org
DS=$DS python3 - <<'PY' > /tmp/met_keys_$DS.txt
import urllib.request, urllib.parse, re, os
DS = os.environ['DS']
keys=[]; marker=""
while True:
    u="https://s3.amazonaws.com/openneuro.org/?list-type=2&prefix="+DS+"/&max-keys=1000"+(("&continuation-token="+urllib.parse.quote(marker,safe="")) if marker else "")
    x=urllib.request.urlopen(u).read().decode()
    keys+=re.findall(r"<Key>([^<]+)</Key>",x)
    m=re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>",x)
    if not m: break
    marker=m.group(1)
print("\n".join(keys))
PY
wc -l /tmp/met_keys_$DS.txt
cat /tmp/met_keys_$DS.txt | DS=$DS xargs -P 256 -I{} sh -c 'f="{}"; f=${f#$DS/}; mkdir -p "$(dirname "$f")"; [ -s "$f" ] || { curl -sfL --retry 5 "'$B'/$DS/$f" -o "$f.part" && mv "$f.part" "$f"; }'
du -sh .; find . -type f | wc -l
