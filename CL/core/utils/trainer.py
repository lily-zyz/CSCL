import time
import pdb

# numpy
import numpy as np

# torch
import torch # type: ignore

# internal
from core.utils.utils import set_model_precision, smooth_rank_measure

# global magic numbers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HE_POSITION = 0 # HE slide is always the first one 
WHOLE_VIEW_POSITION = 0


# move to utils
def calculate_losses(STAINS, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, criterionASP, epoch, low_feature, high_feature, modality_labels_withoutHE, args):
    '''
    loss_fn_interMod (torch.nn.Module): The inter-modality loss function.(InfoNCE)
    loss_fn_interMod_local (torch.nn.Module): The local alignment loss function.(GOT)
    loss_fn_intraMod (torch.nn.Module): The intra-modality loss function.(InfoNCE)
    '''
    # torch.Size([30, 5, 2048, 512])
    losses = []
    atleast_two_loss_flag = False
    # ['HE', 'HER2', 'PGR', 'KI67', 'ER']

    # pdb.set_trace()
    for stain_idx, stain in enumerate(STAINS):
        stain_idx+=1
        low_HE_tokens = low_feature[:, HE_POSITION, :, :]
        high_HE_tokens = high_feature[:, HE_POSITION, :, :]
        low_IHC_tokens = low_feature[:, stain_idx, :, :]
        high_IHC_tokens = high_feature[:, stain_idx, :, :]
        # if loss_fn_interMod:
            # print(f"wsi_embs['HE'].shape: {wsi_embs['HE'].shape}") # torch.Size([82, 3, 512, 4])
            # print(f"wsi_embs[stain].shape: {wsi_embs[stain].shape}") # torch.Size([82, 3, 512])
            # print(f"stain_mask.shape: {stain_mask.shape}") #  torch.Size([82])
            # print(f"wsi_embs: {wsi_embs}")# 字典{HE: tensor([[[...]]], stain: tensor([[[...]]])}
            # print(f"WHOLE_VIEW_POSITION: {WHOLE_VIEW_POSITION}") # 0
            # print(f"stain_idx: {stain_idx}") # 0 1 2 3
            # if args.global_loss == "info-nce":
            #     low_global_loss = loss_fn_interMod(query=low_HE_tokens, positive_key=low_IHC_tokens, symmetric=args.symmetric_cl)
            #     high_global_loss = loss_fn_interMod(query=high_HE_tokens, positive_key=high_IHC_tokens, symmetric=args.symmetric_cl)
            #     global_loss = low_global_loss + high_global_loss
            # else:
            #     raise AssertionError("invalid global loss")

            # # add to loss 
            # losses.append(global_loss) 
        
        # Local loss:
        if loss_fn_interMod_local:
            # print(f"HE_tokens.shape: {HE_tokens.shape}")
            # print(f"IHC_tokens.shape: {IHC_tokens.shape}")
            # HE_tokens.shape: torch.Size([90, 2048, 128])
            # IHC_tokens.shape: torch.Size([90, 2048, 128])
            low_got_loss = loss_fn_interMod_local(low_HE_tokens, low_IHC_tokens, subsample=256)
            high_got_loss = loss_fn_interMod_local(high_HE_tokens, high_IHC_tokens, subsample=256)
            got_loss = low_got_loss + high_got_loss
            got_loss = got_loss * args.local_loss_weight

            # add to loss 
            losses.append(got_loss)
        # ASP loss
        if criterionASP:
            # HE_tokens = token_embs["HE"][:, :, :, stain_idx][stain_mask]
            # IHC_tokens = token_embs[stain].squeeze()[stain_mask]
            low_ASP_loss = criterionASP(low_HE_tokens, low_IHC_tokens, current_epoch=epoch)
            high_ASP_loss = criterionASP(high_HE_tokens, high_IHC_tokens, current_epoch=epoch)
            ASP_loss = (low_ASP_loss+high_ASP_loss)*10.0
            losses.append(ASP_loss)
        # there is at least one stain in addition to HE in this batch, so we keep this batch
        atleast_two_loss_flag = True
            
    if len(losses) > 0:
        loss = sum(losses)
    else:
        loss = -1
        assert loss == -1 and not atleast_two_loss_flag, "Loss should be -1 if there are no losses to calculate"
        
    return loss, atleast_two_loss_flag

# move to utils
def train_loop(args, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, criterionASP, ssl_model, epoch, dataloader, optimizer, scheduler_warmup, scheduler):

    if loss_fn_intraMod:
        n_views = 3
    else:
        n_views = 1
        
    ssl_model.train()
    torch_precision = set_model_precision(args.precision)

    ep_loss = 0.
    fb_time = 0.
    # all_embeds = []
    
    for b_idx, data in enumerate(dataloader):
        '''
        data: dict
            - feats: tensor, [80, 5, 2048, 512]
            - modality_labels: tensor, [80, 5]
            - slide_ids: list, len=80
        '''
        # print(f"b_idx: {b_idx}")
        # print(f"data: {data}")
        if epoch == 0 and b_idx == 0:
            print("Using precision:", torch_precision)
        
        s_fb = time.time()
        
        # clean modality labels to be without HE
        modality_labels = data['modality_labels']
        modality_labels_withoutHE = modality_labels[:, HE_POSITION+1:]
        
        # begin forward pass
        optimizer.zero_grad()
             
        with torch.amp.autocast(device_type="cuda", dtype=torch_precision):
            
            # get model outputs
            # print(f"data: {data}")
            low_feature, high_feature = ssl_model(data=data, device=DEVICE, n_views=n_views)
        
            # calculate losses
            loss, atleast_two_loss_flag = calculate_losses(args.STAINS, loss_fn_interMod, loss_fn_interMod_local, loss_fn_intraMod, criterionASP, epoch, low_feature, high_feature, modality_labels_withoutHE, args)
            
        # get the train embeds to calculate rank
        # all_embeds.extend(wsi_embs['HE'][:, WHOLE_VIEW_POSITION, :, 0].detach().to(torch.float32).cpu().numpy())
        
        # if we have a batch with only HE then continue
        if not atleast_two_loss_flag:
            print("Skipping batch with only HE")
            continue
        
        # if we have made it, then we must have had more than HE stain, so we update model
        loss.backward()
        optimizer.step()

        if epoch <= args.warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler.step()  
            
        if (b_idx % 3) == 0:
            print(f"Loss for batch: {b_idx} = {loss:.3f}")
            
        ep_loss += loss.item()
        
        e_fb = time.time()
        fb_time += e_fb - s_fb
        
    # track rank on all HE slides
    # all_embeds_tensor = torch.Tensor(np.array(all_embeds))
    # rank = smooth_rank_measure(all_embeds_tensor)  
    
    return ep_loss
    # return ep_loss, rank