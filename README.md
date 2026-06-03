# Backbone Classification Project

Train M1 image classification with a shared training loop and switchable backbone.

```bash
cd /sdb1/liran/downsteam_code/classify/classify_backbone_project
python train.py --backbone usfm --epochs 50 --batch_size 8 --lr 1e-4
python train.py --backbone fetalclip --epochs 50 --batch_size 8 --lr 1e-4
python train.py --backbone dinov3 --epochs 50 --batch_size 8 --lr 1e-4
python train.py --backbone openus --epochs 50 --batch_size 8 --lr 1e-4
```

The dataset reader only requires `split` and `image_path`. Labels are inferred from the parent folder of `image_path`, for example `.../M1/0101/image.png -> 0101`.

Backbone code for USFM and FetalCLIP is vendored directly in `models/usfm.py` and `models/fetalclip.py`, so the classification project does not import implementation files from `compare_code/`. DINOv3 and OpenUS still use their local repo paths because the upstream model implementations are much larger.

Outputs are saved under `runs/<backbone>-EXP-*`: `log.txt`, `metrics.csv`, training curves, best confusion matrices, `checkpoint/best_model.pth`, and `checkpoint/last_model.pth`.
