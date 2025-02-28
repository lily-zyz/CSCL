#!/bin/bash

## Global and Local loss with stain encodings pretraining ###
CUDA_VISIBLE_DEVICES=2,4,6 python pretrain.py \
    --high_data_root_dir /mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT_Wintraloss/results/ACROBAT/processing_slides_4_mag_10x_patchsize_256/patch_embeddings \
    --low_data_root_dir /mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT_Wintraloss/results/ACROBAT/processing_slides_4_mag_5x_patchsize_256/patch_embeddings \
    --results_dir results_brca \
    --cohort brca \
    --dataset ACROBAT \
    --csv_fpath /mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT_Wintraloss/dataset_csv/ACROBAT/ACROBAT2.csv \
    --wsi_encoder abmil \
    --n_heads 4 \
    --patch_embedding_dim 1024 \
    --wsi_encoder_hidden_dim 512 \
    --global_loss "info-nce" \
    --local_loss "got" \
    --intra_modality_loss "info-nce" \
    --local_loss_weight 1.0 \
    --temperature 0.001 \
    --lr 0.0001 \
    --max_epochs 120 \
    --batch_size 30 \
    --num_gpus 3 \
    --opt adamW \
    --num_workers 20 \
    --n_subsamples 512 \
    --activation softmax \
    --warmup_epochs 5 \
    --warmup \
    --symmetric_cl \
    --add_stain_encoding \
    --precision bfloat16
