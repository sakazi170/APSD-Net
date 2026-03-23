# Enhancing Multi-Modal Brain Tumor Segmentation via Asymmetric Primary-Secondary Dynamic Network
## Usage
### Data Preparation
Please download BraTS 2020 data according to https://www.med.upenn.edu/cbica/brats2020/data.html.
### Training
#### Training on the entire BraTS training set
```bash
python train.py --model BTS_t2 --mixed --trainset
```
#### Breakpoint continuation for training
```bash
python train.py --model BTS_t2 --mixed --trainset --cp checkpoint
```
### Inference
```bash
python test.py --model BTS_t2 --labels --cp checkpoint
```
