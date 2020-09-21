#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnet18 \
#     --ckpt_file pl_ckpts/model-resnet18-albu_re-specific_mixup/best-model.pt

# CUDA_VISIBLE_DEVICES=0 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model se_resnext101_32x4d \
#     --ckpt_file pl_ckpts/model-se_resnext101_32x4d-albu_re/best-model.pt


# CUDA_VISIBLE_DEVICES=2 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/model-resnest101-albu_re/best-model.pt

# CUDA_VISIBLE_DEVICES=2 python predict_ml.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/ml-model-resnest101-albu_re/best-model.pt


# CUDA_VISIBLE_DEVICES=2 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/model-resnest101-albu_re-autoaug-ricap/best-model.pt
    # --ckpt_file pl_ckpts/model-resnest101-albu_re-ricap/best-model.pt
    # --ckpt_file pl_ckpts/model-resnest101-albu_re-autoaug/best-model.pt

# CUDA_VISIBLE_DEVICES=2 python predict.py \
#     --root /share/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/model-resnest101-albu_re-pseudo_90/best-model.pt    

    # --ckpt_file pl_ckpts/model-resnest101-albu_re-pseudo_95_pseudo_95/best-model.pt    


    # --ckpt_file pl_ckpts/model-resnest101-albu_re-pseudo_99/best-model.pt    

    # --ckpt_file pl_ckpts/model-resnest101-albu_re-pseudo_95/best-model.pt    

# CUDA_VISIBLE_DEVICES=2 python predict_pil.py \
#     --root /data/Dataset/plant-pathology-2020-fgvc7 \
#     --model resnest101 \
#     --ckpt_file pl_ckpts/pil-model-resnest101-albu_re/best-model.pt

# pseudo label
# python pseudo_label.py --save_file pseudo_99.csv --prob_thr 0.99
# python pseudo_label.py --save_file pseudo_98.csv --prob_thr 0.98
# python pseudo_label.py --save_file pseudo_95.csv --prob_thr 0.95
# python pseudo_label.py --save_file pseudo_95_pseudo_95.csv --prob_thr 0.95
# python pseudo_label.py --save_file pseudo_90.csv --prob_thr 0.90

CUDA_VISIBLE_DEVICES=4 python predict.py \
    --root /data/hejy/hejy_dp/datasets/ICPR2020_CHARTINFO_UB_PMC_TRAIN \
    --model resnest101 \
    --ckpt_file /data/hejy/hejy_dp/cls_he/pl_ckpts/pil-model-resnest101-albu_re_autoaug/best-model.pt 

