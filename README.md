# CSCL

# Cross-Stain Contrastive Learning for Paired Immunohistochemistry and Histopathology Slide Representation ( CSCL)
> We provide bash script for training and inference on Our dataset
> Before running the code, ensure you have downloaded these dataset and preprocess following [CONCH](https://github.com/mahmoodlab/CONCH)
> Our dataset can be downloaded from <a href=https://www.anonymz.com/?https://huggingface.co/datasets/zyzzyzyz/HER2/tree/main>HER2</a>

The list of useful parameters is as follows:
* `local_dir`: this is the path where you put your '.pt' files for patch features.
* `model_dir`: the location of CS or CL model weights.
* `slide_embedding_pkl`: the way to slide embedding.
* `label_path`: the way to test dataset label CSV file.

### Train Adapter (CL)
```bash
cd CL
cd ./bin
bash ../scripts/train_adapter.sh
```
### Using Adapter for Patch Embedding
```bash
bash extract_Adapter_embedding.sh
```

### Train CS
```bash
cd CS
cd ./bin
bash ../scripts/launch_pretrain_withStainEncodings.sh
```

### Using CS for Slide Embedding
```
cd ./bin
python extract_slide_embeddings.py
```

### Inference
```
python run_linear_probing.py
```
