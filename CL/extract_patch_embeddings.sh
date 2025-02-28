# python extract_patch_embeddings.py \
#     --slide_dir /mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/data/BCNB/WSIs \
#     --local_dir ../results/BCNB \
#     --patch_mag 10 \
#     --patch_size 256
python extract_BCNB_embeddings.py \
    --patch_dir /mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/data/BCNB/patches \
    --local_dir ../results/BCNB \
    --patch_mag 10 \
    --patch_size 256