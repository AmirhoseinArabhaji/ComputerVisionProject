# GeoClip V1 vs V2 Training and Evaluation Summary

This document summarizes the changes and reported results from the four notebooks:

- [GeoClip_Training_V1.ipynb](GeoClip_Training_V1.ipynb)
- [GeoClip_Training_V2.ipynb](GeoClip_Training_V2.ipynb)
- [GeoClip_Evaluation_V1.ipynb](GeoClip_Evaluation_V1.ipynb)
- [GeoClip_Evaluation_V2.ipynb](GeoClip_Evaluation_V2.ipynb)

## High-Level Changes

V2 is a more developed training and evaluation pass than V1.

- Training V1 uses the standard fine-tuning setup with a frozen GeoCLIP backbone plus a small trainable vision adaptation stack and regression head.
- Training V2 adds PEFT and LoRA support, applies LoRA adapters to the vision transformer attention layers, and keeps the domain-adaptation layers and regressor trainable.
- Training V2 also improves checkpoint handling, including clearer resume logic for pre-LoRA and LoRA-compatible checkpoints.
- Evaluation V2 points to the newer checkpoint directory and reports results from the V2-trained model.

## Training Results

| Notebook | Main training setup | Best validation loss | Best saved checkpoint error |
|---|---|---:|---:|
| Training V1 | Standard GeoCLIP fine-tuning with the last 2 vision transformer layers unfrozen | 0.16167 | 165.2 km |
| Training V2 | LoRA-enhanced fine-tuning with PEFT, LoRA rank 8, alpha 16, and LoRA dropout 0.05 | 0.11837 | 121.0 km |

### Training V1

- Uses checkpoints_v5.
- Keeps the GeoCLIP backbone mostly frozen and trains the last 2 vision transformer layers, visual_projection, image_encoder.mlp, and the regression head.
- Uses a combined loss of L1 plus Haversine distance.
- The notebook finished with a best validation loss of 0.16167.
- The best saved model checkpoint was logged at about 165.2 km validation error.

### Training V2

- Uses checkpoints_v6.
- Installs PEFT and applies LoRA to the vision transformer attention projections (q_proj and v_proj).
- Keeps the backbone in eval mode while training the LoRA adapters, projection layer, MLP, and regression head.
- Adds more explicit checkpoint compatibility handling for old checkpoints and LoRA checkpoints.
- The notebook finished with a best validation loss of 0.11837.
- The best saved model checkpoint was logged at about 121.0 km validation error.

## Evaluation Results

### Evaluation V1

Evaluation V1 loads the checkpoints_v5 model and compares baseline GeoCLIP with the fine-tuned model.

- Baseline GeoCLIP: mean error 617.1 km, median error 216.0 km.
- Fine-tuned GeoClipItaly: mean error 177.3 km, median error 168.0 km.

### Evaluation V2

Evaluation V2 loads the checkpoints_v6 model and uses the same evaluation flow.

- Baseline GeoCLIP: mean error 1015.8 km, median error 244.7 km.
- Fine-tuned GeoClipItaly: mean error 157.0 km, median error 103.3 km.

## What Improved in V2

- Better training efficiency and flexibility through LoRA-based fine-tuning.
- Stronger validation performance, with best validation loss dropping from 0.16167 to 0.11837.
- Better best-checkpoint error, dropping from about 165.2 km to about 121.0 km.
- Better fine-tuned evaluation median error in the V2 evaluation run, dropping from 168.0 km to 103.3 km.

## Notes

- Both notebook versions keep the same core Italy dataset split logic, with the same random seed and split ratios.
- Both evaluation notebooks include the same charting, mapping, and MC Dropout sections; the main difference is the checkpoint path and the model version being loaded.
- The V2 notebooks are the recommended baseline for any further training or evaluation work.