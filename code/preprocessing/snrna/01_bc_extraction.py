
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

import subprocess
import os


@click.command()
@click.argument("fastq-r1", type=click.Path(exists=True))
@click.argument("fastq-merged", type=click.Path(exists=True))
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
    "--constant-sequence-wpre",
    default="GATACCGAGCGCTGC",
    help="Constant sequence that should precede the tag",
)
@click.option(
    "--constant-sequence-polya",
    default="TCGAGAGATCTACGGG",
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
@click.option(
    "--run-merge",
    default=True,
    help="Number of mismatches to allow in tag",
    show_default=True,
)
@click.option(
    "--merge-path",
    default="/broad/macosko/mkim/fake.fastq.gz",
    help="Path for output",
)
@click.option("--debug", is_flag=True, help="Turn on debug logging")
def main(
    fastq_r1,
    fastq_merged,
    output_dir,
    tag_sequence,
    constant_sequence_wpre,
    constant_sequence_polya,
    tag_mismatch,
    const_mismatch,
    run_merge,
    merge_path,
    debug=False,
):


    try:
        os.mkdir(output_dir)
    except:
        print('Path already exists')

    output_dir = Path(output_dir)
    print(f"Saving output to {output_dir}")

    command = [f'zcat {fastq_r1} | head -n 2 | tail -n 1 | wc -c']
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    rd1_length = result.stdout.strip()

  
    print("Fuzzy grep-ing for WPRE constant sequence")
    print(constant_sequence_wpre)
    command = [f'zcat merged.fastq.gz | /usr/bin/ug -Z3 \"{constant_sequence_wpre}\" | pigz > merged_fuzzyMatch.fastq.gz']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True, text=True)

    print("Fuzzy grep-ing for PolyA constant sequence")
    print(constant_sequence_polya)
    command = [f'zcat merged_fuzzyMatch.fastq.gz | /usr/bin/ug -Z3 \"{constant_sequence_polya}\" | pigz > merged_fuzzyMatch2.fastq.gz']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True, text=True)

    start_id=int(rd1_length)+26
    end_id=int(rd1_length)+57
    print("Cutting out the barcodes/identifiers/UMIs")
    command = [f'zcat merged_fuzzyMatch2.fastq.gz | cut -c {start_id}-{end_id} | pigz > merged_fuzzyMatch_notMatch_cellID.fastq.gz; zcat merged_fuzzyMatch2.fastq.gz | cut -c 1-16 | pigz > merged_fuzzyMatch_notMatch_cellBC.fastq.gz; zcat merged_fuzzyMatch2.fastq.gz | cut -c 17-28 | pigz > merged_fuzzyMatch_notMatch_UMI.fastq.gz']
    result = subprocess.run(command, cwd=output_dir, shell=True, capture_output=True, text=True)



if __name__ == "__main__":
    main()

