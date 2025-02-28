import sys; sys.path.append('../')
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
EXTENSIONS = ['.svs', '.mrxs', '.tiff', '.tif', '.TIFF', '.ndpi', '.jpg']

def process(patch_dir, out_dir, patch_mag, patch_size):
    # 获取所有子文件夹
    slide_folders = [f for f in os.listdir(patch_dir) if os.path.isdir(os.path.join(patch_dir, f))]
    logger.info(f'Found {len(slide_folders)} slide folders.')

    # 创建输出目录
    out_dir = os.path.join(out_dir, 'processing_slides_{}_mag_{}x_patchsize_{}'.format(
        len(slide_folders),
        patch_mag,
        patch_size
    ))
    
    # 由于已经有patches，不需要分割和patch目录
    patch_emb_path = os.path.join(out_dir, 'patch_embeddings')
    os.makedirs(patch_emb_path, exist_ok=True)

    # 初始化embedder
    embedder = TileEmbedder(target_patch_size=patch_size, target_mag=patch_mag, save_path=out_dir)

    # 遍历每个slide文件夹
    for slide_folder in tqdm(slide_folders, desc='Processing slides'):
        slide_path = os.path.join(patch_dir, slide_folder)
        
        # 获取该slide下的所有patch文件
        patches = [f for f in os.listdir(slide_path) if any(f.endswith(ext) for ext in EXTENSIONS)]
        if not patches:
            logger.warning(f'No patches found in folder: {slide_folder}')
            continue
            
        logger.info(f'Processing {len(patches)} patches for slide: {slide_folder}')
        
        # 直接对patches进行embedding计算
        embedder.embed_tiles_patches(
            patches_path=slide_path,
            fn=slide_folder,
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
