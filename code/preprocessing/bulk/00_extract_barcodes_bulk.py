import gzip
import itertools
import logging
from collections import Counter, defaultdict
from pathlib import Path
import pickle

import click
import matplotlib
import matplotlib.colors
import networkx as nx
import numpy as np
import scipy.io
import scipy.sparse
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import radius_neighbors_graph

# from slideseq.util.logger import create_logger

import subprocess
import os

log = logging.getLogger("barcode_matrix")

# WSBWSDWSHWSVWSBWSWSHWSVWSBWSDWSH

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

REVERSE_COMP_DICT = str.maketrans('ACGTN', 'TGCAN')

def match_tags(
    cell_ID, umis, barcode_codes, barcode_cutoff
):
    no_id_struct = 0
    pass_all = 0

    raw_umis_per_tag = defaultdict(set)
    raw_reads_per_umi = defaultdict(Counter)

    for tag, umi in zip(cell_ID, umis):

        # Skip if Cell ID does not match structure
        if sum(b in s for b, s in zip(tag, barcode_codes)) < barcode_cutoff:
            no_id_struct += 1
            continue

        pass_all += 1

        raw_umis_per_tag[tag].add(umi)
        raw_reads_per_umi[tag][umi] += 1

    log.debug(f"IDs w/o correct ID structure: {no_id_struct}")
    log.debug(f"Passed all: {pass_all}")

    raw_umis_per_tag = {
        tag_seq: len(umi_seq) for tag_seq, umi_seq in raw_umis_per_tag.items()
    }

    return raw_umis_per_tag, raw_reads_per_umi


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

def write_matrix_1d(upb, beads, output_file):
    m = scipy.sparse.dok_matrix((len(beads)), dtype=np.int32)
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
@click.option(
    "--tag-sequence",
    default="WSBWSDWSHWSVWSBWSWSHWSVWSBWSDWSH",
    help="Format for the tag sequence",
)

@click.option(
    "--constant-sequence-pretag",
    default="TGCCACCCGTAGATCTCTCGA",
    help="Constant sequence that should precede the tag",
)
@click.option(
    "--constant-sequence-posttag",
    default="CGATACCGAGCGCTGCACCGG",
    help="Constant sequence that should follow the tag",
)
@click.option(
    "--tag-mismatch",
    type=int,
    default=3,
    help="Number of mismatches to allow in tag",
    show_default=True,
)
@click.option(
    "--const-mismatch",
    type=int,
    default=3,
    help="Number of mismatches to allow in tag",
    show_default=True,
)
@click.option("--debug", is_flag=True, help="Turn on debug logging")
def main(
    fastq_r1,
    fastq_r2,
    output_dir,
    tag_sequence,
    constant_sequence_pretag,
    constant_sequence_posttag,
    tag_mismatch,
    const_mismatch,
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
    log.info(f"Saving output to {output_dir}")

    command = [f'zcat {fastq_r1} | head -n 2 | tail -n 1 | wc -c']
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    rd1_length = result.stdout.strip()

    rd1_length=int(rd1_length)
    a=rd1_length-1
    log.info(f'Read 1 is {a} nt long')


    log.info("Saving only the sequence lines from read 1")
    command = [f'zcat {fastq_r1} | awk \'NR%4==2\' > rd1_stripped.fastq']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True)

    log.info("Saving only the sequence lines from read 2")
    command = [f'zcat {fastq_r2} | awk \'NR%4==2\' > rd2_stripped.fastq']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True)

    log.info("Merging Read 1 and Read 2 files")
    command = [f'paste rd1_stripped.fastq rd2_stripped.fastq | gzip > merged.fastq.gz']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True)


    log.info("Fuzzy grep-ing for polyA constant sequence")
    print(constant_sequence_pretag)
    command = [f'zcat merged.fastq.gz | /broad/macosko/mkim/Scripts/ugrep/bin/ug -Z3 \"{constant_sequence_pretag}\" | gzip > merged_preFilt.fastq.gz']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True, text=True)

    log.info("Fuzzy grep-ing for WPRE constant sequence")
    print(constant_sequence_posttag)
    command = [f'zcat merged_preFilt.fastq.gz | /broad/macosko/mkim/Scripts/ugrep/bin/ug -Z3 \"{constant_sequence_posttag}\" | gzip > merged_preFilt_postFilt.fastq.gz']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True, text=True)

    bc_start = rd1_length+26
    bc_end = bc_start+31
  
    log.info("Cutting out the identifiers/UMIs")
    command = [f'zcat merged_preFilt_postFilt.fastq.gz | cut -c 1-10 | gzip > umi_preFilt_postFilt.fastq.gz; zcat merged_preFilt_postFilt.fastq.gz | cut -c {bc_start}-{bc_end} | gzip > ident_preFilt_postFilt.fastq.gz']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True, text=True)

    log.info(f"Reading identifiers/UMIs")
    cellID_path = output_dir / 'ident_preFilt_postFilt.fastq.gz'
    umi_path = output_dir / 'umi_preFilt_postFilt.fastq.gz'
    with gzip.open(cellID_path, "rt") as fh:
        cell_ID = [line.strip() for line in itertools.islice(fh, 0, None, 1)]
    with gzip.open(umi_path, "rt") as fh:
        umis = [line.strip() for line in itertools.islice(fh, 0, None, 1)]

    assert len(cell_ID) == len(umis), "read different number of reads"

    log.info(f"Total of {len(cell_ID)} reads")

    barcode_codes = [DEGENERATE_BASE_DICT[b] for b in tag_sequence]
    barcode_cutoff = len(tag_sequence) - tag_mismatch

    log.info(f"barcode_cutoff: {barcode_cutoff}")

    raw_umis_per_tag, raw_reads_per_umi = match_tags(
        cell_ID, umis, barcode_codes, barcode_cutoff
    )

    log.info("Writing output files")
    raw_tags = sorted(raw_umis_per_tag.keys())
    umis_count = [str(raw_umis_per_tag[t]) for t in raw_tags]
    raw_umis = sorted({u for t in raw_tags for u in raw_reads_per_umi[t]})

    log.info(f"{len(raw_tags)} unique bead IDs")

    with gzip.open(output_dir / "raw_tags.txt.gz", "wt") as out:
        print("\n".join(raw_tags), file=out)
    with gzip.open(output_dir / "umis_per_tags.txt.gz", "wt") as out:
        print("\n".join(umis_count), file=out)
    with gzip.open(output_dir / "raw_umis.txt.gz", "wt") as out:
        print("\n".join(raw_umis), file=out)

    log.debug("Writing raw umi/reads matrix")
    write_matrix(raw_reads_per_umi, raw_tags, raw_umis, output_dir / "raw_read_umi_matrix.mtx.gz")
    log.info("Done!")


if __name__ == "__main__":
    main()


