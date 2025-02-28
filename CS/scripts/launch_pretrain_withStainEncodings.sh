#!/bin/bash

## Global and Local loss with stain encodings pretraining ###
CUDA_VISIBLE_DEVICES=1,4,7 python pretrain.py \
    --data_root_dir /mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT_VIT_adapter/results/ACROBAT/high_patch_embeddings \
    --results_dir results_color_attention_no_add \
    --cohort brca \
    --dataset ACROBAT \
    --csv_fpath /mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT/dataset_csv/ACROBAT/ACROBAT2.csv \
    --wsi_encoder abmil \
    --n_heads 4 \
    --patch_embedding_dim 512 \
    --wsi_encoder_hidden_dim 512 \
    --global_loss "info-nce" \
    --local_loss "got" \
    --intra_modality_loss "info-nce" \
    --local_loss_weight 1.0 \
    --temperature 0.001 \
    --lr 0.0001 \
    --max_epochs 120 \
    --batch_size 24 \
    --num_gpus 3 \
    --opt adamW \
    --num_workers 20 \
    --n_subsamples 2048 \
    --activation softmax \
    --warmup_epochs 5 \
    --warmup \
    --symmetric_cl \
    --add_stain_encoding \
    --precision bfloat16
