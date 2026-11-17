# Checkpoints

Trained arm-(e) detector checkpoints (`trained.pt`, `calibrated.pt`, `qat.pt` from
`python -m splice_det.train`) belong here (gate G04).  They are git-ignored; a release binds
them by SHA-256.  Their integer exports (`export/`) are small and go to `artifacts/trained/<run>/`.  The dev-synthetic manifest used for bring-up
does not depend on any checkpoint.
