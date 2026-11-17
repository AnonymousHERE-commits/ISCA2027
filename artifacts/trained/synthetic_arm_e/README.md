# Synthetic-scene training of arm (e) (provenance S)

Written by `python -m splice_det.toytrain --out artifacts/trained/synthetic_arm_e --train 800
--val 160 --calib 320 --test 160 --epochs 160 --class-epochs 60` (CPU, about 2.5 minutes).
It qualifies the training -> calibration -> export -> integer -> device chain; the scenes have
no image content, so this is not a detector trained on images (that run is the Arrhenius
suite, `detector/hpc/README.md`).  The forecast here is the integer reference's Eq. (3),
which carries no exposure discount yet.

| file | content |
|---|---|
| `manifest.model.json` | model manifest extending `contracts/manifest.dev.json`: calibrated band exponents, gate thresholds, partner prior, model constants (provenance S, non-deployable) |
| `generated/` | sources generated from that manifest (`tools/generate_sources.py`); training and the slice check ran under `generated/py` |
| `tensors.svb` | integer heads of the toy model (`heads.*`, int64 container of int8 values), SVB1 format (`reference/splicev_ref/svb.py`) |
| `heads.json` | the same heads as JSON (input of `toytrain.slice_check`) |
| `float_tensors.npz` | the initial (untrained) float maps, for comparison |
| `inventory.json` | every exported tensor with its shape and whether it was trained |
| `calibration.json` | the calibration record (gate pairs, prior counts, band exponents) |
| `report.json` | residual statistics before/after training on the test scenes, slice check, timings |
| `train_log.jsonl` | every logged step |
| `export_report.json` | SHA-256 of the exported files |

Results on the 160 test scenes: residual r = dH - dHhat 895 LSB^2 without a forecast, 395
LSB^2 with the initial maps, 175 LSB^2 with the trained maps (|r| >= 16 LSB: 37% -> 17%); the
trained heads run bit-exactly through the AIE units and PL engines of the toy slice (4/4
scenes).  `tests/test_detector_train.py` re-checks the hashes, the contract and the slice.
