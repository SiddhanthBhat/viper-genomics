import pysam
import numpy as np
from encoding import one_hot_encode

def extract_windows(vcf_path, fasta_path, window=101):
    vcf = pysam.VariantFile(vcf_path)
    fasta = pysam.FastaFile(fasta_path)
    X, y = [], []

    half = window // 2

    for rec in vcf:
        if list(rec.filter.keys()) != ["PASS"]:
            continue

        pos = rec.pos - 1
        seq = fasta.fetch(rec.chrom, max(0, pos-half), pos+half+1)
        if len(seq) != window:
            continue

        X.append(one_hot_encode(seq))
        y.append(int("TIER=1" in str(rec.info)))

    return np.array(X), np.array(y)
