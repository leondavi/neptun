# DNBN vs YOLO — STL-10 Classification Results

## Experiment Overview

This experiment compares the DNBN 8-node cooperative system against YOLOv8s-cls and YOLO11s-cls (YOLO-small classification variants) on STL-10 image classification. The goal is to evaluate how cooperative multi-model communication scales to higher-resolution images (96×96) with limited labeled training data (5,000 samples).

**Dataset:** STL-10 (96×96 RGB, 10 classes, 5,000 train / 8,000 test)
**Device:** Apple MPS (Mac Mini)
**Comparison method:** Same training pipeline (Adam optimizer, cross-entropy loss), same data splits. All models trained from scratch (no pretrained weights).

---

## YOLOv8s-cls Baseline Sweep

YOLOv8s-cls is the classification variant of the YOLO-small detection architecture. It uses a CSPDarknet backbone with a classification head. The model was trained from scratch on 96×96 STL-10 images.

| Config | Epochs | LR | Test Accuracy | Test F1 | Params | Time |
|--------|--------|----|---------------|---------|--------|------|
| yolov8s_e10_lr0.001 | 10 | 0.001 | 0.5659 | 0.5644 | 5,093,546 | 291s |
| yolov8s_e20_lr0.001 | 20 | 0.001 | 0.5828 | 0.5766 | 5,093,546 | 570s |
| **yolov8s_e30_lr0.001** | **30** | **0.001** | **0.5929** | **0.5940** | **5,093,546** | **841s** |
| yolov8s_e20_lr0.0005 | 20 | 0.0005 | 0.5753 | 0.5811 | 5,093,546 | 557s |

**Best YOLOv8s result:** 59.29% accuracy, 59.40% F1 at 30 epochs with lr=0.001.

---

## YOLO11s-cls Baseline Sweep

YOLO11s-cls is the classification variant of YOLO11-small, a newer architecture with 86 layers vs YOLOv8's 56 layers. Also trained from scratch.

| Config | Epochs | LR | Test Accuracy | Test F1 | Params | Time |
|--------|--------|----|---------------|---------|--------|------|
| yolo11s_e10_lr0.001 | 10 | 0.001 | 0.4850 | 0.4654 | 5,455,818 | 288s |
| yolo11s_e20_lr0.001 | 20 | 0.001 | 0.5751 | 0.5742 | 5,455,818 | 577s |
| **yolo11s_e30_lr0.001** | **30** | **0.001** | **0.5931** | **0.5936** | **5,455,818** | **885s** |
| yolo11s_e20_lr0.0005 | 20 | 0.0005 | 0.5705 | 0.5730 | 5,455,818 | 587s |

**Best YOLO11s result:** 59.31% accuracy, 59.36% F1 at 30 epochs with lr=0.001.

Note: YOLOv8s and YOLO11s achieve nearly identical best accuracy (~59.3%) on STL-10. YOLO11s has 7% more parameters but does not improve performance at this resolution and data scale.

---

## DNBN 8-Node Neuron Sweep

DNBN 8-node systems with ring topology were trained for 10 epochs with cosine LR scheduling. Communication neuron sizes (M = C) were swept from 48 down to 16.

| M/C | Test Accuracy | Test F1 | Params | vs Best YOLO Acc | vs Best YOLO F1 |
|-----|---------------|---------|--------|------------------|-----------------|
| **48** | **0.5428** | **0.5281** | **2,002,801** | **−5.03%** | **−6.59%** |
| 16 | 0.5139 | 0.4842 | 1,677,553 | −7.92% | −10.98% |
| 24 | 0.5080 | 0.4917 | 1,739,665 | −8.51% | −10.23% |
| 32 | 0.4980 | 0.4744 | 1,814,577 | −9.51% | −11.96% |

**Best DNBN result:** M48/C48 with 54.28% accuracy, 52.81% F1 (2.00M params).

---

## Model Complexity Comparison

| Model | Architecture | Params | Accuracy | F1 | Params/Accuracy |
|-------|-------------|--------|----------|-----|-----------------|
| YOLOv8s-cls (30ep) | CSPDarknet + Classify | 5,093,546 | 59.29% | 59.40% | 85,912 |
| YOLO11s-cls (30ep) | YOLO11 Backbone + Classify | 5,455,818 | 59.31% | 59.36% | 91,990 |
| DNBN M48/C48 (8-node, 10ep) | 8× ConvNet + GRU comm | 2,002,801 | 54.28% | 52.81% | 36,904 |

DNBN achieves **2.3× better parameter efficiency** (params per accuracy point) compared to YOLOv8s-cls, despite lower absolute accuracy.

---

## Analysis: Why YOLO Leads on STL-10

Unlike CIFAR-10 (where DNBN outperformed YOLO by 4.7pp), on STL-10 the YOLO baselines outperform DNBN by ~5pp. Several factors explain this:

1. **Limited training data:** STL-10 has only 5,000 labeled training images (vs 50,000 for CIFAR-10). DNBN's 8-node ensemble has more parameters to coordinate but each expert sees the same small dataset, making it harder for the communication protocol to develop complementary specializations.

2. **Higher resolution benefits YOLO:** STL-10 images are 96×96 — much closer to YOLO's design target (224×224+). YOLO's multi-stage downsampling backbone is well-suited at this resolution, whereas on 32×32 CIFAR-10 it destroyed spatial features too aggressively.

3. **Epoch disparity:** YOLO baselines were trained for up to 30 epochs while DNBN ran for 10 epochs. DNBN's per-epoch training cost is higher due to communication rounds, so extending training could narrow the gap.

4. **Parameter efficiency still favors DNBN:** Despite lower accuracy, DNBN M48 uses only 2.00M params (2.5× fewer than YOLO's 5.09M), achieving 54.3% vs 59.3%. The params-per-accuracy-point ratio favors DNBN (36,904 vs 85,912).

---

## Training Configuration

### YOLO Baselines
- Optimizer: Adam, lr=0.001 (and 0.0005 sweep)
- No LR scheduler
- Batch size: 64
- Data augmentation: random horizontal flip + normalization

### DNBN 8-Node
- Optimizer: Adam, lr=0.001, weight_decay=1e-4
- LR scheduler: cosine annealing
- Batch size: 64
- Communication rounds: 5
- Topology: ring (bidirectional)
- Bond sparsity lambda: 0.0
- Gradient clipping: 1.0
- Epochs: 10

---

## Conclusion

On STL-10 (96×96, 5K training images), both YOLOv8s-cls and YOLO11s-cls reach ~59.3% accuracy at 30 epochs, while the best DNBN 8-node configuration (M48/C48) achieves 54.3% at 10 epochs with 2.5× fewer parameters. The limited training set size and higher image resolution favor YOLO's monolithic backbone over DNBN's distributed expert approach on this benchmark. DNBN continues to offer superior parameter efficiency (2.3× better params-per-accuracy-point), suggesting that with more training data or longer training, the cooperative multi-model approach could close the gap.
