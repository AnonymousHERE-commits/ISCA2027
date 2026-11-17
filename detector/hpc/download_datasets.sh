#!/bin/bash
# Datasets and pretrained weights of the SPLICE experiments, placed where the Arrhenius
# launcher (run_splicev.sh) looks for them:
#
#   <base>/datasets/coco/         train2017/  val2017/  annotations/instances_{train,val}2017.json
#   <base>/datasets/crowdhuman/   annotation_train.odgt  annotation_val.odgt  Images/
#   <base>/checkpoints/pretrained/yolo26s.pt
#
# Run it where there is outbound network (a login node; inside tmux or screen, the COCO
# training images alone are about 19 GB):
#
#   bash download_datasets.sh                      # COCO 2017, CrowdHuman and the weights
#   bash download_datasets.sh coco                 # one or more of: coco crowdhuman weights
#
# Running it again continues an interrupted download and skips what is complete.  The
# archives are deleted after a verified extraction (SPLICE_KEEP_ZIPS=1 keeps them).
#
# Sources (override with SPLICE_COCO_URL, SPLICE_CROWDHUMAN_URL, SPLICE_WEIGHTS_URL):
#   COCO 2017    http://images.cocodataset.org  (zips/train2017.zip, zips/val2017.zip,
#                annotations/annotations_trainval2017.zip)
#   CrowdHuman   https://huggingface.co/datasets/sshao0516/CrowdHuman  (CrowdHuman_train01-03.zip,
#                CrowdHuman_val.zip, annotation_train.odgt, annotation_val.odgt); the same files
#                are distributed by https://www.crowdhuman.org - downloaded by hand they can be
#                placed in <base>/datasets/crowdhuman and this script only extracts them
#   YOLO26-S     https://github.com/ultralytics/assets/releases (yolo26s.pt, release v8.4.0)
set -euo pipefail

BASE="${SPLICE_BASE:-/nobackup/proj/disk/naiss2025-6-428/from-alvis/SPLICE-V}"
DATA_DIR="${BASE}/datasets"
COCO_ROOT="${SPLICE_COCO_ROOT:-${DATA_DIR}/coco}"
CROWDHUMAN_ROOT="${SPLICE_CROWDHUMAN_ROOT:-${DATA_DIR}/crowdhuman}"
WEIGHTS="${SPLICE_WEIGHTS:-${BASE}/checkpoints/pretrained/yolo26s.pt}"
KEEP_ZIPS="${SPLICE_KEEP_ZIPS:-0}"
COCO_URL="${SPLICE_COCO_URL:-http://images.cocodataset.org}"
CROWDHUMAN_URL="${SPLICE_CROWDHUMAN_URL:-https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main}"
WEIGHTS_URL="${SPLICE_WEIGHTS_URL:-https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt}"
# Sizes of the complete splits (a mirror with fewer images is reported, not accepted)
N_COCO_TRAIN="${SPLICE_N_COCO_TRAIN:-118287}"
N_COCO_VAL="${SPLICE_N_COCO_VAL:-5000}"
N_CROWD_TRAIN="${SPLICE_N_CROWD_TRAIN:-15000}"
N_CROWD_VAL="${SPLICE_N_CROWD_VAL:-4370}"

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '[FATAL] %s\n' "$*" >&2; exit 1; }

# fetch URL DEST: resumable download through a .part file
fetch() {
  local url="$1" dst="$2"
  mkdir -p "$(dirname "${dst}")"
  if [[ -s "${dst}" ]]; then say "have $(basename "${dst}")"; return 0; fi
  say "downloading ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -c --tries=5 --timeout=60 --no-verbose --show-progress -O "${dst}.part" "${url}" || die "download failed: ${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 10 -C - -o "${dst}.part" "${url}" || die "download failed: ${url}"
  else
    die "neither wget nor curl is available on this host"
  fi
  [[ -s "${dst}.part" ]] || die "empty download: ${url}"
  mv "${dst}.part" "${dst}"
}

# extract ZIP DESTDIR (files that already exist are kept)
extract() {
  local zip="$1" dst="$2"
  mkdir -p "${dst}"
  say "extracting $(basename "${zip}")"
  if command -v unzip >/dev/null 2>&1; then
    unzip -q -n "${zip}" -d "${dst}" || die "extraction failed: ${zip} (delete it and run again)"
  else
    python3 -m zipfile -e "${zip}" "${dst}" || die "extraction failed: ${zip} (delete it and run again)"
  fi
}

count_images() {
  local d="$1"
  [[ -d "${d}" ]] || { echo 0; return 0; }
  find "${d}" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l
}

drop_zip() { if [[ "${KEEP_ZIPS}" != 1 ]]; then rm -f "$1"; fi; }

get_coco() {
  local name want n
  mkdir -p "${COCO_ROOT}"
  for name in train2017 val2017; do
    if [[ "${name}" == train2017 ]]; then want="${N_COCO_TRAIN}"; else want="${N_COCO_VAL}"; fi
    n="$(count_images "${COCO_ROOT}/${name}")"
    if (( n == want )); then say "COCO ${name}: complete (${n} images)"; continue; fi
    say "COCO ${name}: ${n} of ${want} images present"
    fetch "${COCO_URL}/zips/${name}.zip" "${COCO_ROOT}/${name}.zip"
    extract "${COCO_ROOT}/${name}.zip" "${COCO_ROOT}"
    n="$(count_images "${COCO_ROOT}/${name}")"
    (( n == want )) || die "COCO ${name}: ${n} images after extraction, expected ${want} (archive kept: ${COCO_ROOT}/${name}.zip)"
    drop_zip "${COCO_ROOT}/${name}.zip"
    say "COCO ${name}: complete (${n} images)"
  done
  if [[ -s "${COCO_ROOT}/annotations/instances_train2017.json" && -s "${COCO_ROOT}/annotations/instances_val2017.json" ]]; then
    say "COCO annotations: complete"
  else
    fetch "${COCO_URL}/annotations/annotations_trainval2017.zip" "${COCO_ROOT}/annotations_trainval2017.zip"
    extract "${COCO_ROOT}/annotations_trainval2017.zip" "${COCO_ROOT}"
    [[ -s "${COCO_ROOT}/annotations/instances_train2017.json" && -s "${COCO_ROOT}/annotations/instances_val2017.json" ]] \
      || die "COCO annotations: instances_train2017.json / instances_val2017.json missing after extraction"
    drop_zip "${COCO_ROOT}/annotations_trainval2017.zip"
    say "COCO annotations: complete"
  fi
}

# number of annotated images of an odgt file that exist under Images/
crowd_present() {
  local odgt="$1"
  python3 - "${odgt}" "${CROWDHUMAN_ROOT}" <<'PYCOUNT'
import json, os, sys
odgt, root = sys.argv[1], sys.argv[2]
have = set()
for dirpath, _, files in os.walk(root):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            have.add(os.path.splitext(f)[0])
with open(odgt) as f:
    ids = [json.loads(line)["ID"] for line in f if line.strip()]
print(sum(1 for i in ids if i in have), len(ids))
PYCOUNT
}

get_crowdhuman() {
  local f part present total
  mkdir -p "${CROWDHUMAN_ROOT}"
  for f in annotation_train.odgt annotation_val.odgt; do
    fetch "${CROWDHUMAN_URL}/${f}" "${CROWDHUMAN_ROOT}/${f}"
  done
  for part in CrowdHuman_train01 CrowdHuman_train02 CrowdHuman_train03 CrowdHuman_val; do
    if [[ -f "${CROWDHUMAN_ROOT}/.extracted_${part}" ]]; then say "CrowdHuman ${part}: extracted"; continue; fi
    fetch "${CROWDHUMAN_URL}/${part}.zip" "${CROWDHUMAN_ROOT}/${part}.zip"
    extract "${CROWDHUMAN_ROOT}/${part}.zip" "${CROWDHUMAN_ROOT}"
    : > "${CROWDHUMAN_ROOT}/.extracted_${part}"
    drop_zip "${CROWDHUMAN_ROOT}/${part}.zip"
  done
  read -r present total < <(crowd_present "${CROWDHUMAN_ROOT}/annotation_train.odgt")
  (( present == total && total == N_CROWD_TRAIN )) \
    || die "CrowdHuman train: ${present} of ${total} annotated images on disk, expected ${N_CROWD_TRAIN} (remove ${CROWDHUMAN_ROOT}/.extracted_* and run again)"
  say "CrowdHuman train: complete (${present} images)"
  read -r present total < <(crowd_present "${CROWDHUMAN_ROOT}/annotation_val.odgt")
  (( present == total && total == N_CROWD_VAL )) \
    || die "CrowdHuman val: ${present} of ${total} annotated images on disk, expected ${N_CROWD_VAL} (remove ${CROWDHUMAN_ROOT}/.extracted_* and run again)"
  say "CrowdHuman val: complete (${present} images)"
}

get_weights() {
  fetch "${WEIGHTS_URL}" "${WEIGHTS}"
  say "weights: ${WEIGHTS} ($(du -h "${WEIGHTS}" | cut -f1), sha256 $(sha256sum "${WEIGHTS}" | cut -c1-16)...)"
}

[[ -d "${BASE}" ]] || die "project folder not found: ${BASE} (set SPLICE_BASE)"
what=("$@")
(( ${#what[@]} > 0 )) || what=(coco crowdhuman weights)
say "target: ${BASE}"
df -h "${BASE}" 2>/dev/null | sed 's/^/           /' || true
for item in "${what[@]}"; do
  case "${item}" in
    coco) get_coco ;;
    crowdhuman) get_crowdhuman ;;
    weights) get_weights ;;
    *) die "unknown item '${item}' (use: coco crowdhuman weights)" ;;
  esac
done
say "done.  Check everything before submitting:  bash ${BASE}/code/run_splicev.sh check"
