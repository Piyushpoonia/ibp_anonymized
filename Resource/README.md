# Frozen resources

`obj_enc.pth` is the PointNet object encoder checkpoint used by the feature
extractor. `hf_models.part01` through `hf_models.part03` are consecutive
1 GiB-or-smaller chunks of the original `hf_models.tar.gz` cache archive,
containing CLIP ViT-B/16 and BLIP image-captioning base. Do not extract
individual chunks. Run `python resource/install_weights.py` from the package
root; it verifies the concatenated archive SHA-256
`5e9e1a2f942fbdb61065d21c0b9cd7dc600bcad204aa7035f5bf4db422aaae85`
and installs the cache under `resource/hf_home/hub`.

If these files were cloned through Git LFS, run `git lfs pull` before installing.
Model weights have upstream licenses. Verify their redistribution terms before
publishing this folder publicly.
