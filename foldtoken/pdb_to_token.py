# inference.py: Encode a PDB structure into tokens using FoldToken4
import os
import torch
import argparse
import tqdm
import csv
from omegaconf import OmegaConf
from model_interface import MInterface
from src.chroma.data import Protein


def load_model(config_path, checkpoint_path, device='cpu'):
    """Load the FoldToken4 model from checkpoint on given device"""
    print(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)

    print("Initializing model...")
    model = MInterface(**config)

    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Using device: {device}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    # adjust keys if wrapped in DataParallel
    for key in list(checkpoint.keys()):
        if '_forward_module.' in key:
            checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
            del checkpoint[key]

    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    print("Model loaded successfully")
    return model


def encode_pdb_to_tokens(model, pdb_path, device='cpu', level=8):
    """Encode a PDB file into tokens using FoldToken4 on given device"""
    protein = Protein(pdb_path, device=device)
    # determine number of residues via XCS representation
    X, C, S = protein.to_XCS()

    with torch.no_grad():
        vq_code = model.encode_protein(protein, level=level)[1]
        return vq_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encode all PDBs in a directory to tokens CSV.")
    parser.add_argument('--pdb_dir', required=True, help="Directory containing PDB files.")
    parser.add_argument('--output', default="output.csv", help="Output CSV file path.")
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu', help="Compute device to use.")
    args = parser.parse_args()

    # Validate device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Config and checkpoint paths
    config_path = os.path.join(os.path.dirname(__file__), 'checkpoint', 'FT4', 'config.yaml')
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint', 'FT4', 'ckpt.pth')
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(config_path, checkpoint_path, device)

    # Gather PDB files
    pdb_files = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir) if f.lower().endswith('.pdb')]
    results = []
    for pdb_path in tqdm.tqdm(pdb_files, desc="Encoding PDBs", unit="file"):
        basename = os.path.basename(pdb_path)
        try:
            protein = Protein(pdb_path, device=device)
            X, C, S = protein.to_XCS()
        except Exception as e:
            print(f"Skipped {basename}: load error {e}")
            continue
        # filter short or highly masked proteins
        if X.shape[1] < 5 or (C != -1).sum() < 5:
            print(f"Skipped {basename}: too few residues")
            continue
        with torch.no_grad():
            vq_code = model.encode_protein(protein, level=8)[1]
        token_list = vq_code.cpu().numpy().tolist() if hasattr(vq_code, 'cpu') else list(vq_code)
        results.append((basename, token_list))

    # write results
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pdb_file', 'tokens'])
        for name, tlist in results:
            writer.writerow([name, ' '.join(map(str, tlist))])
    print(f"Processed {len(results)}/{len(pdb_files)} files. Tokens saved to {args.output}")

