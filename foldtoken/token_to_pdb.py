import os
import torch
import argparse
import csv

import tqdm
from omegaconf import OmegaConf
from model_interface import MInterface

# Load FoldToken4 model for decoding

def load_model(device='cpu'):
    cwd = os.path.dirname(__file__)
    config_path = os.path.join(cwd, 'checkpoint', 'FT4', 'config.yaml')
    checkpoint_path = os.path.join(cwd, 'checkpoint', 'FT4', 'ckpt.pth')
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    model = MInterface(**config)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    # adjust keys if wrapped in DataParallel
    for key in list(checkpoint.keys()):
        if '_forward_module.' in key:
            checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()
    return model

# Convert token list to PDB and save

def tokens_to_pdb(model, tokens, output_path, level=8):
    device = next(model.parameters()).device
    vq_codes = torch.tensor(tokens, dtype=torch.long, device=device)
    # get latent embeddings
    h_V = model.model.vq.embed_id(vq_codes, level)
    # simple chain encoding
    chain_encoding = torch.ones_like(vq_codes, device=device)
    # decode to protein object
    protein = model.model.decoding(h_V, chain_encoding)
    # save PDB
    protein.to(output_path)

# CLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decode tokens CSV to PDB files.")
    parser.add_argument('--token_csv', required=True, help="CSV file with columns ['pdb_file','tokens'].")
    parser.add_argument('--output_dir', required=True, help="Directory to save reconstructed PDBs.")
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu', help="Compute device.")
    parser.add_argument('--level', type=int, default=8, help="Quantization level.")
    args = parser.parse_args()

    # Decoding currently requires CUDA-enabled build
    if not torch.cuda.is_available():
        print("Error: decoding tokens requires CUDA-enabled PyTorch. Please install the GPU version and run with --device cuda.")
        exit(1)
    device = 'cuda'
    if args.device == 'cuda' and device == 'cpu':
        print("CUDA not available, using CPU which may not support decoding.")
    elif args.device == 'cpu' and device == 'cuda':
        print("CUDA available, overriding to use CUDA for decoding.")

    model = load_model(device)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.token_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in tqdm.tqdm(reader, total=sum(1 for _ in open(args.token_csv))):
            name = row.get('pdb_file') or row.get('pdb')
            tokens = [int(x) for x in row['tokens'].split()]
            out_name = os.path.splitext(name)[0] + '.pdb'
            # Ensure unique filename by appending a counter if needed
            candidate = out_name
            root, ext = os.path.splitext(out_name)
            counter = 1
            while os.path.exists(os.path.join(args.output_dir, candidate)):
                candidate = f"{root}_{counter}{ext}"
                counter += 1
            output_path = os.path.join(args.output_dir, candidate)
            try:
                tokens_to_pdb(model, tokens, output_path, level=args.level)
                # print(f"Saved {output_path}")
            except Exception as e:
                print(f"Failed {name}: {e}")
