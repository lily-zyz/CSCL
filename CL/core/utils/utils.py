# general
import os
from collections import OrderedDict
from tqdm import tqdm
import wandb # type: ignore
import pdb
import argparse
import h5py 
# numpy
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import random

# torch
import torch # type: ignore
import torch.backends.cudnn # type: ignore
import torch.cuda # type: ignore

# internal
from core.utils.file_utils import save_pkl

# global magic numbers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HE_POSITION = 0 # HE slide is always the first one 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_hdf5(output_fpath, 
                  asset_dict, 
                  attr_dict= None, 
                  mode='a', 
                  auto_chunk = True,
                  chunk_size = None):
    """
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            # 确保数据是numpy数组，并处理bfloat16类型
            if torch.is_tensor(val):
                val = val.to(torch.float32).detach().cpu().numpy()
            
            # 修改这部分代码，增加安全检查
            if key == 'features' and len(val.shape) == 3:
                # 只在第一个维度为1时才进行squeeze操作
                if val.shape[0] == 1:
                    val = val.squeeze(0)
                else:
                    print('output_fpath:',output_fpath)
                    pdb.set_trace()
                    # 如果第一个维度不为1，直接使用第一个样本
                    val = val[0]
            
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f:
                data_type = val.dtype
                if data_type == np.object_: 
                    data_type = h5py.string_dtype(encoding='utf-8')
                if auto_chunk:
                    chunks = True
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(key, 
                                          shape=data_shape, 
                                          chunks=chunks,
                                          maxshape=(None,) + data_shape[1:],
                                          dtype=data_type)
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except Exception as e:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5: {str(e)}")
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    
    return output_fpath

# move to utils
def run_inference(ssl_model, val_dataloader, config=None, torch_precision=None, save_path=None):
    """
    Perform validation loop for the SSL model.

    Args:
        config (object): Configuration object containing model settings.
        ssl_model (object): SSL model to be evaluated.
        val_dataloader (object): Dataloader for validation dataset.

    Returns:
        tuple: A tuple containing the results dictionary and the rank measure.
            - results_dict (dict): Dictionary containing the embeddings and slide IDs.
            - rank (float): Rank measure calculated from the embeddings.
    """

    # set model to eval 
    ssl_model.eval()
    if torch_precision is None:
        torch_precision = set_model_precision(config.precision)
    # all_embeds = []
    # all_slide_ids = []
    all_low_feature = []
    all_high_feature = []
    # do everything without grads
    # embedding_save_path = os.path.join(self.save_path, 'patch_embeddings', f'{fn}.h5')
    with torch.no_grad():
        # for batch in tqdm(val_dataloader):
        #     print("Batch contents:", batch)
        i = 0
        for feats, slide_ids, low_coords, high_coords in tqdm(val_dataloader):
            # forward
            with torch.amp.autocast(device_type="cuda", dtype=torch_precision):
                low_feature, high_feature = ssl_model.get_features(feats, device=DEVICE)
                all_low_feature.extend(low_feature.to(torch.float32).detach().cpu().numpy())
                all_high_feature.extend(high_feature.to(torch.float32).detach().cpu().numpy())
            current_slide_id = slide_ids[0]
            
            # 创建保存路径
            low_embed_dir = os.path.join(save_path, 'low_patch_embeddings')
            high_embed_dir = os.path.join(save_path, 'high_patch_embeddings')
            
            # 创建文件夹（如果不存在）
            os.makedirs(low_embed_dir, exist_ok=True)
            os.makedirs(high_embed_dir, exist_ok=True)
            
            # 构建完整的文件保存路径
            low_embedding_save_path = os.path.join(low_embed_dir, f'{current_slide_id}.h5')
            high_embedding_save_path = os.path.join(high_embed_dir, f'{current_slide_id}.h5')
            
            mode = 'w' if i == 0 else 'a'

            original_low_feature = low_feature[0]
            low_asset_dict = {
                'features': original_low_feature,
                'coords': low_coords[0],
            }
            high_asset_dict = {
                'features': high_feature,
                'coords': high_coords[0],
            }
            i += 1
            # pdb.set_trace()
            save_hdf5(low_embedding_save_path, mode=mode, asset_dict=low_asset_dict)
            save_hdf5(high_embedding_save_path, mode=mode, asset_dict=high_asset_dict)
                # print(f"low_feature: {low_feature.shape}, high_feature: {high_feature.shape}")
                # low_feature: torch.Size([1, 332, 512]), high_feature: torch.Size([1, 332, 512])
                # pdb.set_trace()            
                # all_embeds.extend(wsi_embed.to(torch.float32).detach().cpu().numpy())
                # all_slide_ids.append(slide_ids[0])
            
    # all_embeds = np.array(all_embeds)
    # all_embeds_tensor = torch.Tensor(all_embeds)
    # rank = smooth_rank_measure(all_embeds_tensor)  
    # results_dict = {"embeds": all_embeds, 'slide_ids': all_slide_ids}
    
    return all_low_feature, all_high_feature

def extract_slide_level_embeddings(args, val_dataloaders, ssl_model):
    """
    Extracts slide-level embeddings for each dataset in val_dataloaders using the provided ssl_model.

    Args:
        args (object): The arguments object containing various configuration options.
        val_dataloaders (dict): A dictionary containing the validation dataloaders for each dataset.
        ssl_model (object): The SSL model used for extracting embeddings.

    Returns:
        None
    """
    for dataset_name in val_dataloaders:
        print(f"\n* Extracting slide-level embeddings of {dataset_name}")
        curr_loader = val_dataloaders[dataset_name]
        curr_results_dict, curr_val_rank = run_inference(ssl_model, curr_loader, config=args)
        print("Rank for {} = {}".format(dataset_name, curr_val_rank))
        print("\033[92mDone \033[0m")
        
        if args.log_ml:
            wandb.run.summary["{}_rank".format(dataset_name)] = curr_val_rank
            
        save_pkl(os.path.join(args.RESULS_SAVE_PATH, f"{dataset_name}.pkl"), curr_results_dict)

def load_checkpoint(args, ssl_model, path_to_checkpoint=None):
    """
    Loads a checkpoint file and updates the state of the SSL model.

    Args:
        args (Namespace): The command-line arguments.
        ssl_model (nn.Module): The SSL model to update.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint file is corrupted or incompatible with the model.

    """
    # load checkpoint
    if path_to_checkpoint is not None:
        state_dict = torch.load(path_to_checkpoint)
    else:
        state_dict = torch.load(os.path.join(args.RESULS_SAVE_PATH, "model.pt"))
    
    # load weights into model
    try:
        ssl_model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        ssl_model.load_state_dict(new_state_dict)
        print('Model loaded by removing module in state dict...')
    
    return ssl_model

def set_model_precision(precision):
    """
    Sets the precision of the model to the specified precision.

    Args:
        model (torch.nn.Module): The model to set the precision for.
        precision (str): The desired precision. Can be one of 'float64', 'float32', or 'bfloat16'.

    Returns:
        tuple: A tuple containing the model with the updated precision and the corresponding torch precision.
    """
    if precision == 'float64':
        torch_precision = torch.float64
    elif precision == 'float32':
        torch_precision = torch.float32
    elif precision == 'bfloat16':
        torch_precision = torch.bfloat16
    else:
        raise ValueError(f"Invalid precision: {precision}")
    
    return torch_precision


def set_deterministic_mode(SEED, disable_cudnn=False):
    """
    Sets the random seed for various libraries to ensure deterministic behavior.

    Args:
        SEED (int): The seed value to use for random number generation.
        disable_cudnn (bool, optional): Whether to disable cuDNN. Defaults to False.

    Notes:
        - Sets the random seed for torch, random, numpy, and torch.cuda.
        - If `disable_cudnn` is False, also sets cuDNN to use deterministic algorithms.
        - If `disable_cudnn` is True, disables cuDNN.

    """
    torch.manual_seed(SEED)  # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)  # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED)))  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = False  # Causes cuDNN to deterministically select an algorithm,
        # possibly at the cost of reduced performance
        # (the algorithm itself may be nondeterministic).
        torch.backends.cudnn.deterministic = True  # Causes cuDNN to use a deterministic convolution algorithm,
        # but may slow down performance.
        # It will not guarantee that your training process is deterministic
        # if you are using other libraries that may use nondeterministic algorithms
    else:
        torch.backends.cudnn.enabled = False  # Controls whether cuDNN is enabled or not.
        # If you want to enable cuDNN, set it to True.
    

def smooth_rank_measure(embedding_matrix, eps=1e-7):
    """
    Compute the smooth rank measure of a matrix of embeddings.
    
    Args:
        embedding_matrix (torch.Tensor): Matrix of embeddings (n x m). n: number of patch embeddings, m: embedding dimension
        alpha (float): Smoothing parameter to avoid division by zero.

    Returns:
        float: Smooth rank measure.
    """
    
    # Perform SVD on the embedding matrix
    _, S, _ = torch.svd(embedding_matrix)
    
    # Compute the smooth rank measure
    p = S / torch.norm(S, p=1) + eps
    p = p[:embedding_matrix.shape[1]]
    smooth_rank = torch.exp(-torch.sum(p * torch.log(p)))
    smooth_rank = round(smooth_rank.item(), 2)
    
    return smooth_rank
