import sys; sys.path.append('../'); sys.path.append('./')
import argparse
import os
import logging

import openslide
from tqdm import tqdm

from core.preprocessing.conch_patch_embedder import TileEmbedder
from core.preprocessing.hest_modules.segmentation import TissueSegmenter
from core.preprocessing.hest_modules.wsi import get_pixel_size, OpenSlideWSI

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# File extensions for slide images
EXTENSIONS = ['.svs', '.mrxs', '.tiff', '.tif', '.TIFF', '.ndpi', '.png']

def process(patch_dir, out_dir, patch_mag, patch_size):
    # 获取所有子文件夹（每个子文件夹代表一个modality）
    modality_folders = [f for f in os.listdir(patch_dir) if os.path.isdir(os.path.join(patch_dir, f))]
    logger.info(f'Found {len(modality_folders)} modality folders.')

    # 创建输出目录
    out_dir = os.path.join(out_dir, 'processing_slides_{}_mag_{}x_patchsize_{}'.format(
        len(modality_folders),
        patch_mag,
        patch_size
    ))
    
    # 由于已经有patches，不需要分割和patch目录
    patch_emb_path = os.path.join(out_dir, 'patch_embeddings')
    os.makedirs(patch_emb_path, exist_ok=True)

    # 初始化embedder
    embedder = TileEmbedder(target_patch_size=patch_size, target_mag=patch_mag, save_path=out_dir)

    # 遍历每个modality文件夹
    for modality_folder in tqdm(modality_folders, desc='Processing modalities'):
        modality_path = os.path.join(patch_dir, modality_folder)
        
        # 处理trainA (HE) 和 trainB (特定modality)
        for train_folder in ['trainA', 'trainB']:
            train_path = os.path.join(modality_path, train_folder)
            if not os.path.exists(train_path):
                logger.warning(f'Train folder not found: {train_path}')
                continue
                
            # 确定modality
            modality = 'HE' if train_folder == 'trainA' else modality_folder
            
            # 获取该目录下的所有图片
            patches = [f for f in os.listdir(train_path) if any(f.endswith(ext) for ext in EXTENSIONS)]
            if not patches:
                logger.warning(f'No patches found in folder: {train_path}')
                continue
                
            # 按slide_id分组处理patches
            slide_patches = {}
            for patch in patches:
                slide_id = patch.split('_')[0]
                if slide_id not in slide_patches:
                    slide_patches[slide_id] = []
                slide_patches[slide_id].append(patch)
            
            # 处理每个slide的patches
            for slide_id, patch_list in slide_patches.items():
                logger.info(f'Processing {len(patch_list)} patches for slide {slide_id} ({modality})')
                
                # 设置输出文件名，移除'train'后缀
                output_name = f'{slide_id}_{modality}'
                
                # 进行embedding计算
                embedder.embed_ACROBAT_patches(
                    patches_path=train_path,
                    fn=output_name,
                    patch_list=patch_list,
                    slide_id=slide_id,
                    modality=modality
                )

    logger.info('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--slide_dir", type=str, help="Directory with slides.", default=None)
    parser.add_argument("--patch_dir", type=str, help="Directory with slides.", default=None)
    parser.add_argument("--local_dir", type=str, help="Where to save tissue segmentation, patch coords, and patch embeddings.", default='./../data/downstream')
    parser.add_argument("--patch_mag", type=int, help="Magnification at which patching operates. Default to 10x.", default=10)
    parser.add_argument("--patch_size", type=int, help="Patch size. Default to 256.", default=256)

    args = parser.parse_args()

    logger.info('Initiate run...')
    process(args.patch_dir, args.local_dir, args.patch_mag, args.patch_size)
