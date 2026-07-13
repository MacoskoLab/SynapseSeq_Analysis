import gzip
import itertools
import logging
from collections import Counter, defaultdict
from pathlib import Path
import pickle

import click
import matplotlib
import matplotlib.colors

import subprocess
import os

log = logging.getLogger("barcode_matrix")

DEGENERATE_BASE_DICT = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "M": {"A", "C"},
    "K": {"G", "T"},
    "S": {"C", "G"},
    "W": {"A", "T"},
    "H": {"A", "C", "T"},
    "B": {"C", "G", "T"},
    "V": {"A", "C", "G"},
    "D": {"A", "G", "T"},
    "N": {"A", "C", "G", "T"},
}

def match_tags(
    cell_ID, cell_BC, umis, barcode_codes, barcode_cutoff
):
    no_id_struct = 0
    pass_all = 0

    raw_umis_per_bead = defaultdict(lambda: defaultdict(set))
    raw_umis_per_bead_umi = defaultdict(Counter)

    for tag, bead_bc, umi in zip(cell_ID, cell_BC, umis):

        # Skip if Cell ID does not match structure
        if sum(b in s for b, s in zip(tag, barcode_codes)) < barcode_cutoff:
            no_id_struct += 1
            continue

        pass_all += 1

        bead_umi_bc = bead_bc + umi

        raw_umis_per_bead[bead_bc][tag].add(umi)
        raw_umis_per_bead_umi[bead_umi_bc][tag] += 1

    log.debug(f"IDs w/o correct ID structure: {no_id_struct}")
    log.debug(f"Passed all: {pass_all}")

    raw_umis_per_bead = {
        bead_bc: {tag: len(v) for tag, v in tags_per_bead.items()}
        for bead_bc, tags_per_bead in raw_umis_per_bead.items()
    }

    return raw_umis_per_bead, raw_umis_per_bead_umi


def write_matrix(upb, beads, tags, output_file):
    m = scipy.sparse.dok_matrix((len(beads), len(tags)), dtype=np.int32)
    b2i = {b: i for i, b in enumerate(beads)}
    t2j = {t: j for j, t in enumerate(tags)}

    for b, bd in upb.items():
        for t, v in bd.items():
            m[b2i[b], t2j[t]] = v

    m = m.tocsr()

    with gzip.open(output_file, "wb") as out:
        scipy.io.mmwrite(out, m)

    return m

def reverse_h_set(h_barcodes):
    base_d_rev = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

    return ["".join([ base_d_rev[c] for c in h_list ]) for h_list in h_barcodes]

@click.command()
@click.argument("fastq-r1", type=click.Path(exists=True))
@click.argument("fastq-r2", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    help="Path for output",
    required=True,
)

@click.option("--debug", is_flag=True, help="Turn on debug logging")
def main(
    fastq_r1,
    fastq_r2,
    output_dir,
    debug=False,
):
    """
    This script generates some plots for mapping barcoded reads.
    
    Reads sequences from FASTQ_R1 and FASTQ_R2. Assumes that the first read
    contains a 15bp barcode split across two locations, along with an 8bp UMI.
    The second read is assumed to have TAG_SEQUENCE in bases 20-40.
    
    """
    # create_logger(debug, dryrun=False)

    try:
        os.mkdir(output_dir)
    except:
        print('Path already exists')

    output_dir = Path(output_dir)
    print(f"Saving output to {output_dir}")

    command = [f'zcat {fastq_r1} | head -n 2 | tail -n 1 | wc -c']
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    rd1_length = result.stdout.strip()

    rd1_length=int(rd1_length)
    a=rd1_length-1
    print(f'Read 1 is {a} nt long')

    print("Saving only the sequence lines from read 1")
    command = [f'zcat {fastq_r1} | awk \'NR%4==2\' > rd1_stripped.fastq']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True)

    print("Saving only the sequence lines from read 2")
    command = [f'zcat {fastq_r2} | awk \'NR%4==2\' > rd2_stripped.fastq']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True)

    print("Merging Read 1 and Read 2 files")
    command = [f'paste rd1_stripped.fastq rd2_stripped.fastq | pigz > merged.fastq.gz']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True)


if __name__ == "__main__":
    main()
