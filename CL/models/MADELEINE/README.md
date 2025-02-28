---
license: mit
---

# Checkpoint for Multistain Pretraining for Slide Representation Learning in Pathology (ECCV'24)

Welcome to the official HuggingFace repository of the ECCV 2024 paper, "Multistain Pretraining for Slide Representation Learning in Pathology".
This project was developed at the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital.

## Model loging

```
from huggingface_hub import login
login()
```

You can refer [HuggingFace](https://hf-mirror.com/docs/huggingface_hub/en/quick-start#login-command) documentation for more details.

## Preprocessing: tissue segmentation, patching, and patch feature extraction

We are extracting [CONCH](https://github.com/mahmoodlab/CONCH) features at 10x magnification on 256x256-pixel patches. Please refer to CSCL public implementation to extract patch embeddings from a WSI.

## Extracting CSCL slide encoding

You can obtain and run CSCL slide encoding (trained on Acrobat breast samples at 10x magnification) using:

```
from core.models.factory import create_model_from_pretrained

model, precision = create_model_from_pretrained('./models/')

feats = load_h5('your_path_to_conch_patch_embeddings/XXX.h5')

with torch.no_grad():
  with torch.amp.autocast(device_type="cuda", dtype=precision):
      wsi_embed = model.encode_he(feats=feats, device='cuda')
```
