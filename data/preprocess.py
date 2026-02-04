import os
import torch
import pysam
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from encoding import one_hot_encode

def extract_dataset(vcf_path, fasta_path, window=101):
    print(f"Loading VCF: {vcf_path}")
    vcf = pysam.VariantFile(vcf_path)
    fasta = pysam.FastaFile(fasta_path)
    
    X_list, y_list = [], []
    half = window // 2
    
    count = 0
    for rec in vcf:
        # Filter for high-confidence variants
        if list(rec.filter.keys()) != ["PASS"]:
            continue
            
        # Extract Genomic Window
        pos = rec.pos - 1
        chrom = rec.chrom
        
        # Handle chromosome naming mismatches (e.g., '1' vs 'chr1')
        if chrom not in fasta.references:
            if f"chr{chrom}" in fasta.references:
                chrom = f"chr{chrom}"
            else:
                continue

        try:
            seq = fasta.fetch(chrom, max(0, pos-half), pos+half+1)
        except KeyError:
            continue
            
        if len(seq) != window:
            continue

        # One-hot Encode
        encoded_seq = one_hot_encode(seq) # Shape [101, 4]
        
        # Label: Check Info field for 'TIER=1' (CGC Tier 1 Driver)
        # Note: Adjust string matching based on your specific VCF annotation format
        is_driver = 1 if "TIER=1" in str(rec.info) else 0

        X_list.append(encoded_seq)
        y_list.append(is_driver)
        count += 1
        
        if count % 5000 == 0:
            print(f"Processed {count} variants...")

    print(f"Total extracted: {len(X_list)}")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def save_splits(X, y, output_dir, seed=42):
    # Stratified Split: 70% Train, 15% Val, 15% Test
    # First split: Train (70%) vs Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    
    # Second split: Val (15% total) vs Test (15% total) -> 50% of Temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=seed
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save((torch.from_numpy(X_train), torch.from_numpy(y_train)), 
               os.path.join(output_dir, "train.pt"))
    torch.save((torch.from_numpy(X_val), torch.from_numpy(y_val)), 
               os.path.join(output_dir, "val.pt"))
    torch.save((torch.from_numpy(X_test), torch.from_numpy(y_test)), 
               os.path.join(output_dir, "test.pt"))
               
    print(f"Saved splits to {output_dir}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True, help="Path to COSMIC VCF")
    parser.add_argument("--fasta", required=True, help="Path to Reference Genome Fasta")
    parser.add_argument("--out", default="processed_data", help="Output directory")
    args = parser.parse_args()
    
    X, y = extract_dataset(args.vcf, args.fasta)
    save_splits(X, y, args.out)