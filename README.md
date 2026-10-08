# R-CoMBO

**R-CoMBO is an extended version of the CVPR 2025 paper [CoMBO: Conflict Mitigation via Branched Optimization for Class Incremental Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Fang_CoMBO_Conflict_Mitigation_via_Branched_Optimization_for_Class_Incremental_Segmentation_CVPR_2025_paper.html).**

Built on Mask2Former and Detectron2, R-CoMBO retains CoMBO's Query Conflict Reduction (QCR), Half-Learning Half-Distillation (HDHL), and Importance-Based Knowledge Distillation (IKD). It extends the decoder with separate structure and semantic query streams, semantic pixel cross attention, and importance guided semantic gradient routing.


## Environment

The environment and dependency versions below match the inspected `py39` environment:

| Component | Version |
| --- | --- |
| Python | 3.9.23 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| torchaudio | 2.7.1+cu128 |
| CUDA toolkit | 12.8 |
| Detectron2 | 0.6 |
| NumPy | 1.23.5 |
| Pillow | 9.5.0 |
| timm | 1.0.15 |
| OpenCV | 4.11.0.86 |

Additional package versions are pinned in [requirements.txt](requirements.txt). Pillow is pinned to the version actually imported by the environment; its installed package metadata contains a different version.

Run the following commands from the repository root. Set `CUDA_HOME` to your CUDA 12.8 toolkit directory before building the extensions.

```bash
conda create -n rcombo python=3.9.23 pip
conda activate rcombo

python -m pip install pip==25.3 setuptools==59.5.0 wheel==0.45.1
python -m pip install \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

python -m pip install --no-build-isolation \
  git+https://github.com/facebookresearch/detectron2.git@d1e04565d3bec8719335b88be9e9b961bf3ec464
python -m pip install \
  git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0

(cd mask2former/modeling/pixel_decoder/ops && sh make.sh)
```

The Detectron2 and panoptic API revisions above were read from the installed environment's source metadata. A compatible C++ compiler and a CUDA toolkit containing `nvcc` are required. The original experiment used GCC 11.4 and NVIDIA RTX 4090 GPUs.

## Dataset preparation

### ADE20K

Download ADE20K and its instance annotations from the [ADE20K website](http://sceneparsing.csail.mit.edu/). Place the data at `datasets/ADEChallengeData2016`, or create a local symlink there.

```text
datasets/ADEChallengeData2016/
├── images/{training,validation}/
├── annotations/{training,validation}/
├── annotations_instance/{training,validation}/
├── annotations_detectron2/{training,validation}/
├── ade20k_panoptic_{train,val}.json
├── ade20k_panoptic_{train,val}/
└── ade20k_instance_{train,val}.json
```

Generate annotations from the repository root:

```bash
python datasets/prepare_ade20k_sem_seg.py
python datasets/prepare_ade20k_pan_seg.py
python datasets/prepare_ade20k_ins_seg.py
python continual/prepare_datasets.py
```

The last command generates the 100-10, 100-50, 100-5, and 50-50 incremental annotations under `json/pan/`. Semantic experiments reuse these image splits. Keep the bundled static dataset metadata files: the annotation preparation scripts need them. ADE20K training expects the local `datasets/` layout even when other dataset workflows use `DETECTRON2_DATASETS`.

### Cityscapes and Pascal VOC

- **Cityscapes:** place `leftImg8bit/`, `gtFine/`, and prepared panoptic annotations under `datasets/cityscapes/`. Generate a protocol with `python tools/generate_cityscapes_incremental.py --root datasets/cityscapes --base-cls 11 --inc-cls 4`. Corresponding configurations are under `configs/cityscapes/`.
- **Pascal VOC:** prepare VOC2012 and augmented semantic annotations with `datasets/prepare_pascal_voc_sem_seg.py` and `datasets/prepare_continual_voc_sem_seg.py`. These scripts support `VOC_ROOT`, `VOC_TRAIN_LIST`, and `VOC_VAL_LIST`. Use the bundled `incremental_split/` lists or set `VOC_INCREMENTAL_SPLIT_ROOT`. VOC protocol names count foreground classes; configuration class counts also include background.

## Training and evaluation

Train the complete ADE20K 100-10 trajectory with the reference settings:

```bash
bash scripts/train_ade20k_100_10.sh 0,1 output/ps/100-10
```

The launcher uses the active environment's `python`; override it with `PY` if necessary. To start at a later task using existing preceding checkpoints, set `START_TASK`, for example:

```bash
START_TASK=2 bash scripts/train_ade20k_100_10.sh 0,1 output/ps/100-10
```

Evaluate the final task:

```bash
bash scripts/evaluate_ade20k_100_10.sh 0 output/ps/100-10 6
```

For explicit checkpoint evaluation:

```bash
python train_continual.py --num-gpus 1 --eval-only \
  --config-file configs/ade20k/panoptic-segmentation/100-10.yaml \
  CONT.TASK 6 CONT.WEIGHTS ./path/to/model_final.pth \
  OUTPUT_DIR output/ps/100-10/step6
```

Keep trajectory outputs in sibling `step1`, `step2`, ..., `step6` directories. Incremental training reads the preceding task checkpoint; evaluation reads the current task checkpoint. `CONT.WEIGHTS` overrides model initialization and `CONT.OLD_WEIGHTS` sets the teacher checkpoint. To resume an interrupted task, run the training command with `--resume` and the same task output directory and schedule.

## Repository layout

```text
train_continual.py    # Training and evaluation entry point
continual/           # Trainers, dataset mappers, distillation, and importance updates
mask2former/         # Model, decoder, dataset registration, and CUDA operator sources
configs/             # ADE20K, Cityscapes, and VOC configurations
scripts/             # Reference ADE20K 100-10 training and evaluation launchers
datasets/            # Dataset preparation scripts and required static metadata
incremental_split/   # Incremental image lists
tools/               # Dataset generation, conversion, and evaluation utilities
requirements.txt     # Dependencies aligned with the current py39 environment
```

The `.gitignore` excludes local data, generated annotations, logs, checkpoints, caches, and build outputs.

## Citation and acknowledgements

Please cite the original CoMBO paper when building on this work:

```bibtex
@InProceedings{Fang_2025_CVPR,
  author = {Fang, Kai and Zhang, Anqi and Gao, Guangyu and Jiao, Jianbo and Liu, Chi Harold and Wei, Yunchao},
  title = {CoMBO: Conflict Mitigation via Branched Optimization for Class Incremental Segmentation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2025},
  pages = {25667-25676}
}
```

This implementation builds on [Mask2Former](https://github.com/facebookresearch/Mask2Former), [Detectron2](https://github.com/facebookresearch/detectron2), and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
