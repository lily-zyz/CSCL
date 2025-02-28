import argparse
import hashlib
import json
import os
import core.utils.utils as util

def get_args():
    """
    Parse command line arguments for CSCL configurations.
    
    Returns:
        args (argparse.Namespace): Parsed command line arguments.
    """
    print("* Setup args...", end="")
    parser = argparse.ArgumentParser(description='Configurations for CSCL')
    
    #----> set up
    parser.add_argument('--high_data_root_dir', type=str, default="/mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT_Wintraloss/results/ACROBAT/processing_slides_4_mag_10x_patchsize_256/patch_embeddings", help='data directory')
    parser.add_argument('--low_data_root_dir', type=str, default="/mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT_Wintraloss/results/ACROBAT/processing_slides_4_mag_5x_patchsize_256/patch_embeddings", help='data directory')
    parser.add_argument('--results_dir', default='results_brca', help='results directory (default: ../output)')
    parser.add_argument('--cohort', default='brca', help='which disease model do you have')
    parser.add_argument('--dataset', default='ACROBAT', type=str, help='which dataset are you using')
    parser.add_argument('--csv_fpath', type=str, default="/mnt/lpai-dione/ssai/cvg/team/qiuyin/zyz/classification/CSCL_MyACROBAT_Wintraloss/dataset_csv/ACROBAT/ACROBAT2.csv", help='CSV with labels')

    #----> training args
    parser.add_argument('--patch_embedding_dim', type=int, default=1024, help='what is the feature extractor encoding dim')
    parser.add_argument('--max_epochs', type=int, default=120, help='maximum number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, default="adamW", help="Optimizer")
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--n_subsamples', type=int, default=512, help='Number of patches to sample during training.')
    parser.add_argument('--scheduler', type=str, default=None, help='scheduler used for training')
    parser.add_argument('--num_workers', type=int, default=20, help='number of cpu workers')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay in the loss')
    parser.add_argument('--temperature', type=float, default=0.001, help='temperature for training')
    parser.add_argument('--warmup', action='store_true', default=True, help='enable warmup')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='epochs for warmup')
    parser.add_argument('--end_learning_rate', type=float, default=1.0E-8, help='end learning rate')
    parser.add_argument('--num_gpus', type=int, default=3, help='which gpu devices')
    parser.add_argument('--precision', default='bfloat16', help='results directory (default: ../output)')

    #----> model args 
    parser.add_argument('--wsi_encoder', type=str, default="abmil", help='what wsi encoder to use')
    parser.add_argument('--activation', type=str, default="softmax", help='what activation to use')
    parser.add_argument('--wsi_encoder_hidden_dim', type=int, default=512, help='what hidden dim to use')
    parser.add_argument('--n_heads', type=int, default=4, help='number of heads in ABMIL. Only used in SuperSimplifiedMMSSL.')
    parser.add_argument('--add_stain_encoding', action='store_true', default=True, help='should there be stain encodings?')

    #----> loss args 
    parser.add_argument('--symmetric_cl', action='store_true', default=True, help='is loss symmetric?')
    parser.add_argument('--global_loss', type=str, default='info-nce', help='loss used for global alignemnt of different WSI')
    parser.add_argument('--local_loss', type=str, default='got', help='loss used for local alignemnt of different WSI')
    parser.add_argument('--intra_modality_loss', type=str, default='info-nce', help='info-nce loss for comparing different views of same WSI')
    parser.add_argument('--ASP_loss', type=str, default='-1', help='ASP loss for comparing different patches of same WSI')
    parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
    parser.add_argument('--asp_loss_mode', type=str, default='none', help='"scheduler_lookup" options for the ASP loss. Options for both are listed in Fig. 3 of the paper.')
    parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
    parser.add_argument('--local_loss_weight', type=float, default=1.0, help='weight for local loss')
    parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=10, help='number of epochs to linearly decay learning rate to zero')

    #----> log args
    parser.add_argument('--log_ml', action='store_true', help='choose if we want to log results in mlflow and tensorboard')
    parser.add_argument('--wandb_project_name', type=str, default='CSCL', help='Experiment name to use to log to WANDB')
    parser.add_argument('--wandb_entity', type=str, default='CSCL', help='Entity to use to log to WANDB')
    
    #---> model inference 
    parser.add_argument('--pretrained', type=str, default=None, help='Path to dir with checkpoint.')
    
    args = parser.parse_args()
    
    # set paths 
    args.ROOT_SAVE_DIR = "./../{}/".format(args.results_dir)
    args.EXP_CODE = "Cohort:{}_SlideEnc:{}_nHeads:{}_GlobalLoss:{}_LocalLoss:{}_AddSE:{}_LR:{}_Epochs:{}_Batch:{}_nTokens:{}_Temp:{}_Precision:{}".format(
        args.cohort,
        args.wsi_encoder,
        args.n_heads,
        args.global_loss,
        args.local_loss,
        args.add_stain_encoding,
        args.lr, 
        args.max_epochs, 
        args.batch_size, 
        args.n_subsamples,
        args.temperature,
        args.precision,
    )
    
    # convert args namespace to a hash to keep it unique
    args.exp_hash = hashlib.md5(json.dumps({k: str(v) for k, v in vars(args).items()}, sort_keys=True).encode()).hexdigest()
    args.RESULS_SAVE_PATH = os.path.join(args.ROOT_SAVE_DIR, "DEBUG_" + args.exp_hash)
    
    print("\033[92m Done \033[0m")
    
    if args.pretrained is not None:
        print(f"* Running experiment {args.EXP_CODE}...")
    else:
        print(f"* Running inference with model {args.EXP_CODE}...")
        
    
    return args