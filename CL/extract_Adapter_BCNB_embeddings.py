"""
# Usage
python extract_slide_embeddings.py --local_dir ../results/BCNB/ 
"""

# general
import sys; sys.path.append("./"); sys.path.append("./")

import argparse
import os

# internal 
from core.models.factory import create_model_from_pretrained
from core.utils.utils import  run_inference
from torch.utils.data import DataLoader 
from core.utils.file_utils import save_pkl
from core.datasets.wsi_dataset import SimpleDataset, simple_collate, BCNBDataset, BCNB_collate


# define downstream dataset and loader
def get_downstream_loader(high_features_dir, low_features_dir):
    """
    Returns a DataLoader object for downstream dataset.
    Returns:
        DataLoader: A DataLoader object that loads data for downstream processing.
    """
    dataset = BCNBDataset(high_features_dir=os.path.join(high_features_dir), low_features_dir=os.path.join(low_features_dir))
    loader = DataLoader(dataset, num_workers=4, collate_fn=BCNB_collate)     
    return loader


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--high_features_dir", type=str, default="/mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/results/BCNB2/processing_slides_933_mag_10x_patchsize_256")
    parser.add_argument("--low_features_dir", type=str, default="/mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/results/BCNB2/processing_slides_933_mag_5x_patchsize_256")
    parser.add_argument("--save_dir", type=str, default='/mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/results/BCNB2')
    parser.add_argument("--model_dir", type=str, default='/mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/results_brca/global')

    args = parser.parse_args()
    save_dir = args.save_dir
    high_features_dir = args.high_features_dir
    low_features_dir = args.low_features_dir

    # init CSCL model
    model, precision = create_model_from_pretrained(os.path.join(args.model_dir, 'Adapter'))

    # get downstream loader
    dataloader = get_downstream_loader(high_features_dir=high_features_dir, low_features_dir=low_features_dir)

    # extract slide embeddings
    low_feature, high_feature = run_inference(model, dataloader, torch_precision=precision, save_path=save_dir)
    # pdb.set_trace()
    # low_feature, high_feature = results_dict
    # save_pkl(os.path.join(save_dir, "CSCL_slide_embeddings.pkl"), results_dict)
