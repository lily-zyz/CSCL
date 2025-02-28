"""
# Usage
python extract_slide_embeddings.py --local_dir ../results/BCNB/ 
"""

# general
import sys; sys.path.append("../"); sys.path.append("./")

import argparse
import os

# internal 
from core.models.factory import create_model_from_pretrained
from core.utils.utils import  run_inference
from torch.utils.data import DataLoader 
from core.utils.file_utils import save_pkl
from core.datasets.wsi_dataset import SimpleDataset, simple_collate


# define downstream dataset and loader
def get_downstream_loader(path):
    """
    Returns a DataLoader object for downstream dataset.
    Returns:
        DataLoader: A DataLoader object that loads data for downstream processing.
    """
    dataset = SimpleDataset(features_path=os.path.join(path))
    loader = DataLoader(dataset, num_workers=4, collate_fn=simple_collate)     
    return loader


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default='/mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT/results_brca/DEBUG_a93825f6f544b89fb6dc09cb68d5d4e7')

    args = parser.parse_args()
    local_dir = args.local_dir

    # init CSCL model
    model, precision = create_model_from_pretrained(os.path.join(args.model_dir, 'CSCL'))

    # get downstream loader
    dataloader = get_downstream_loader(path=local_dir)

    # extract slide embeddings
    results_dict, rank = run_inference(model, dataloader, torch_precision=precision)
    save_pkl(os.path.join(local_dir, "CSCL_slide_embeddings.pkl"), results_dict)
