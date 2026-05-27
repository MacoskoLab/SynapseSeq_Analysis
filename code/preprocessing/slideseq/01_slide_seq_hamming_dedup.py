"""
Slideseq deduplication using hamming distance
Starts following the barcode extraction step, using the output of 230726_SynapseSeq_Slideseq_Dialout_NoDegen.py
Meaning beads have been grouped, and hamming collapsed.

Results in a deduplicated u and m matrix, as well was reads/umi plots.
"""

import gzip
import itertools
from collections import Counter, defaultdict
import pickle

import csv
import os
import numpy as np
import scipy.io
import scipy.sparse
from sklearn.neighbors import BallTree
import sys

import argparse
from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import multiprocessing as mp
import pandas as pd
sys.path.append("/home/jsilverm/06_synapseseq_repo/synapse_seq_pipeline_code/09_dedup_aav_lib/")
import synapse_seq_functions as ssf
from matplotlib.backends.backend_pdf import PdfPages


def write_mock_fastq(bead_name, umi_vts, size_umi, size_vt, mock_fastq_folder):
    current_fastq_path = os.path.join(mock_fastq_folder, f"{bead_name}.fastq")
    # create mock fastq file
    read_size = size_umi + size_vt
    quality_string = "j" * read_size
    with open(current_fastq_path, "w") as fh:
        for umi_vt in umi_vts:
            umi = umi_vt["umi"]
            vt = umi_vt["vt"]
            count = umi_vt["count"]
            umi_vt = umi + vt
            for i in range(count):
                fh.write(f"@{bead_name}\n")
                fh.write(f"{umi_vt}\n")
                fh.write(f"+\n")
                fh.write(f"{quality_string}\n")
    return current_fastq_path

def run_umi_vt_collapse(bead_name, mock_fastq_folder, output_dir, log_base = "/home/jsilverm/logs"):
    mock_fastq_in = os.path.join(mock_fastq_folder, f"{bead_name}.fastq")
    assert os.path.exists(mock_fastq_in)
    mock_fastq_out = os.path.join(output_dir, f"{bead_name}_out.fastq")
    log_file = os.path.join(log_base, f"{bead_name}_dedup_graph.log")
    cmd = f"umicollapse fastq -k 1 -i {mock_fastq_in} -o {mock_fastq_out} --tag > {log_file} 2>&1"
    os.system(cmd)
    # return output_path

parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input_path', type=str, help='input directory generic')
parser.add_argument("-s", '--input_sample', type=str, help='input sample')
parser.add_argument("-o", '--output_folder', type=str, help='subfolder to write output')

args = parser.parse_args()
input_path = args.input_path
input_sample = args.input_sample
output_folder = args.output_folder


# print all arguments
print('Given arguments:')
print('input_path:', input_path)
print('input_sample:', input_sample)
print('output_folder:', output_folder)

N_CHUNKS_HEAVY = 30
N_CORES_LIGHT = 35
N_CORES_HEAVY = 20

# print global vars
print('N_CHUNKS_HEAVY:', N_CHUNKS_HEAVY)
print('N_CORES_LIGHT:', N_CORES_LIGHT)
print('N_CORES_HEAVY:', N_CORES_HEAVY)

# Hardcoded parameters of read structure
bead_start_inx = 0
bead_end_inx = 14
size_umi = 9
size_vt = 32
vt_hamming_k = 1

curr_path=os.path.join(input_path,input_sample)
output_path=os.path.join(curr_path,output_folder)

if not os.path.exists(output_path):
    os.makedirs(output_path)

intermediate_dir = os.path.join(output_path, "intermediate_dir")
if not os.path.exists(intermediate_dir):
    os.makedirs(intermediate_dir)

# Read in the data
slideseq_dialout={}
with gzip.open(os.path.join(curr_path,"raw_umi_matrix.mtx.gz"), "rb") as fh:
    slideseq_dialout['m'] = scipy.io.mmread(fh).tocsr()
with gzip.open(os.path.join(curr_path,"beads.txt.gz"), "rt") as fh:
    slideseq_dialout['beads'] = [line.strip() for line in fh]
with gzip.open(os.path.join(curr_path,"raw_tags.txt.gz"), "rt") as fh:
    slideseq_dialout['tags'] = [line.strip() for line in fh]
with gzip.open(os.path.join(curr_path,"raw_umi_matrix_umi.mtx.gz"), "rb") as fh:
    slideseq_dialout['u'] = scipy.io.mmread(fh).tocsr()
with gzip.open(os.path.join(curr_path,"beads_umi.txt.gz"), "rt") as fh:
    slideseq_dialout['beads_umi'] = [line.strip() for line in fh]

# Read in the hamming ball tree
aav_ball_tree_obj_path = "/home/jsilverm/06_synapseseq_repo/synapse_seq_pipeline_code/09_dedup_aav_lib/00_hamming_dist_objs/ball_tree_hamming_2.pkl"
with open(aav_ball_tree_obj_path, "rb") as fh:
    ball_tree_obj = pickle.load(fh)

ball_tree = ball_tree_obj['ball_tree']
inx_to_vt_seq_dict = ball_tree_obj['inx_to_seq_dict']


########################################
        # Run VT seq correction
########################################

hamming_correction_results_out_path = os.path.join(intermediate_dir, "hamming_correction_results.pkl")
if os.path.exists(hamming_correction_results_out_path):
    print(f"Loading hamming correction results from {hamming_correction_results_out_path}")
    hamming_correction_results = pickle.load(open(hamming_correction_results_out_path, "rb"))
else:
    chunked_small_list = ssf.chunk_seq_list(slideseq_dialout["tags"], n_chunks=N_CHUNKS_HEAVY)
    radius = vt_hamming_k
    pool = mp.Pool(N_CORES_HEAVY)
    args = [(chunk, ball_tree, radius, inx) for inx, chunk in enumerate(chunked_small_list)]
    hamming_correction_results = pool.starmap(ssf.hamming_correct_seq_chunk, args)
    pool.close()
    pool.join()
    print(f"Writing hamming correction results to {hamming_correction_results_out_path}")
    # write results to file
    with open(hamming_correction_results_out_path, "wb") as fh:
        pickle.dump(hamming_correction_results, fh)

hamming_correction_results_unlisted = list(itertools.chain(*hamming_correction_results))

#Filter criteria:
# VT is not within specified hamming distance of any other VT. Seen as empty lists of neighbors
# VT is greater than 0 hamming distance away from more than 1 VT. (ie it is 1 HD away from 2 different VTs)
parsing_args = [(i, inx_to_vt_seq_dict) for i in hamming_correction_results_unlisted]
parsed_results = [ssf.parse_VT_hamming_result(*arg) for arg in parsing_args]

# Create map of observed VT to whitelist VT
observed_vt_to_whitelist_vt = {}
for i, result in enumerate(parsed_results):
    is_valid = result["is_valid_hamming"]
    if not is_valid:
        continue

    corrected_vt = result["corrected_VT"]
    observed_vt_inx = result["observed_VT_inx"]
    observed_vt_seq = result["observed_VT"]

    h_dist = ssf.do_basic_hamming(observed_vt_seq, corrected_vt)

    observed_vt_to_whitelist_vt[observed_vt_seq] = corrected_vt


# construct umat from corrected VT sequences
u_mat_vt_corrected_data_dict = defaultdict(Counter)
# Find nonzero entries in row.
# For each one, find the correct VT seq it maps to, if any, and add that as continuous sum to u_mat_vt_corrected dict
for i in range(slideseq_dialout["u"].shape[0]):

    bead_umi = slideseq_dialout["beads_umi"][i]
    nonzero_entries = slideseq_dialout["u"][i, :].nonzero()[1]

    for nonzero_entry_inx in nonzero_entries:
        vt_seq = slideseq_dialout["tags"][nonzero_entry_inx]
        corrected_vt = observed_vt_to_whitelist_vt.get(vt_seq, None)
        if corrected_vt is None:
            # this VT is not in the whitelist
            continue
        u_mat_vt_corrected_data_dict[bead_umi][corrected_vt] += slideseq_dialout["u"][i, nonzero_entry_inx]

# create a new u_mat with the corrected VTs
bead_umi_from_corrected_vt = sorted(list(u_mat_vt_corrected_data_dict.keys()))
vts_from_corrected_vt = sorted(set(i for v in u_mat_vt_corrected_data_dict.values() for i in v.keys()))
u_mat_vt_dedup = ssf.create_matrix(u_mat_vt_corrected_data_dict, bead_umi_from_corrected_vt, vts_from_corrected_vt)
vt_dedup_obj = {"u": u_mat_vt_dedup, "bead_umi": bead_umi_from_corrected_vt, "tags": vts_from_corrected_vt}

#########################################################
        # Run UMI deduplication #
    # For each bead, preform graph collapse on umi-VT seq #
#########################################################

# Create mock fastq and output dedup folders
mock_fastq_folder = os.path.join(intermediate_dir, "mock_fastq")
if not os.path.exists(mock_fastq_folder):
    os.makedirs(mock_fastq_folder)

umi_dedup_outdir = os.path.join(intermediate_dir, "umi_dedup_fastqs")
if not os.path.exists(umi_dedup_outdir):
    os.makedirs(umi_dedup_outdir)

# create a dictionary of bead to umi-vt's within it
bead_to_umi_dict = defaultdict(list)
n_rows_u_mat = vt_dedup_obj["u"].shape[0]
for i in range(n_rows_u_mat):
    bead_umi = vt_dedup_obj["bead_umi"][i]
    bead = bead_umi[bead_start_inx:bead_end_inx]
    umi = bead_umi[bead_end_inx:]

    nonzero_elements = vt_dedup_obj["u"][i, :].nonzero()[1]
    for nonzero_element in nonzero_elements:
        vt = vt_dedup_obj["tags"][nonzero_element]
        umi_count = vt_dedup_obj["u"][i, nonzero_element]

        umi_vt = {"umi": umi, "vt": vt, "count": umi_count}
        bead_to_umi_dict[bead].append(umi_vt)

# Create mock fastqs for umicollapse in parallel
create_mock_fastq_args = [(bead_name, umi_vts, size_umi, size_vt, mock_fastq_folder) for bead_name, umi_vts in bead_to_umi_dict.items()]
pool = mp.Pool(N_CORES_LIGHT)
mock_fastq_paths = pool.starmap(write_mock_fastq, create_mock_fastq_args)
pool.close()
pool.join()

# Run umicollapse in parallel
dedup_mock_fastq_args = [(bead_name, mock_fastq_folder, umi_dedup_outdir) for bead_name in bead_to_umi_dict.keys()]
pool = mp.Pool(N_CORES_LIGHT)
dedup_fastq_paths = pool.starmap(run_umi_vt_collapse, dedup_mock_fastq_args)
pool.close()
pool.join()

# Parse the umi collapse results
bead_list = list(bead_to_umi_dict.keys())
parse_umi_dedup_args = [(bead_name, umi_dedup_outdir) for bead_name in bead_list]
pool = mp.Pool(N_CORES_LIGHT)
umi_collapse_results_list = pool.starmap(ssf.parse_umi_collapse_results, parse_umi_dedup_args)
pool.close()
pool.join()

bead_to_umi_dedup = {bead_name: umi_dedup for bead_name, umi_dedup in zip(bead_list, umi_collapse_results_list)}

# create a dictionary of bead-umi to vt, correcting for the deduplication done above
# use the vt_dedup_obj to get the count information

u_mat_dedup_vt_umi_data_dict = defaultdict(Counter)
for i in range(vt_dedup_obj["u"].shape[0]):
    current_bead_umi = vt_dedup_obj["bead_umi"][i]
    bead = current_bead_umi[bead_start_inx:bead_end_inx]
    umi = current_bead_umi[bead_end_inx:]

    bead_umi_vt_deduping = bead_to_umi_dedup.get(bead, None)
    assert bead_umi_vt_deduping is not None

    nonzero_vals = vt_dedup_obj["u"][i, :].nonzero()[1]
    for nonzero_val in nonzero_vals:
        vt = vt_dedup_obj["tags"][nonzero_val]
        count = vt_dedup_obj["u"][i, nonzero_val]

        umi_vt = umi + vt
        umi_vt_correction = bead_umi_vt_deduping.get(umi_vt, None)
        if umi_vt_correction is None:
            print(f"None for {umi_vt} in {bead}")
        assert umi_vt_correction is not None
        umi_corrected = umi_vt_correction[:size_umi]
        vt_corrected = umi_vt_correction[size_umi:]

        bead_umi_corrected = bead + umi_corrected

        u_mat_dedup_vt_umi_data_dict[bead_umi_corrected][vt_corrected] += count

bead_umis_from_dict = sorted(list(u_mat_dedup_vt_umi_data_dict.keys()))
vts_from_dict = sorted(set(i for v in u_mat_dedup_vt_umi_data_dict.values() for i in v.keys()))

u_mat_dedup = ssf.create_matrix(u_mat_dedup_vt_umi_data_dict, bead_umis_from_dict, vts_from_dict)
u_mat_vt_and_umi_dedup_obj = {"u": u_mat_dedup, "bead_umi": bead_umis_from_dict, "tags": vts_from_dict, "bead_split_inx": bead_end_inx}


#########################################################
                # Save Results and plot #
#########################################################

# save the deduplicated u matrix
dedup_u_mat_out_path = os.path.join(output_path, "dedup_vt_and_umi_obj.pkl")
with open(dedup_u_mat_out_path, "wb") as fh:
    pickle.dump(u_mat_vt_and_umi_dedup_obj, fh)

out_pdf_path = os.path.join(output_path, "slideseq_dedup_plots.pdf")

pp = PdfPages(out_pdf_path)

# create reads/umi plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
reads_umi_bins = range(0, 1000, 10)
# create reads/umi plot for the original data
ssf.create_reads_umi_histogram(slideseq_dialout["u"], title_base="Original Data", ax=ax1, bins=reads_umi_bins)
# create reads/umi plot for the VT deduplicated data
ssf.create_reads_umi_histogram(vt_dedup_obj["u"], title_base="VT Deduplication", ax=ax2, bins=reads_umi_bins)
# create reads/umi plot for the VT and UMI deduplicated data
ssf.create_reads_umi_histogram(u_mat_vt_and_umi_dedup_obj["u"], title_base="VT and UMI Deduplication", ax=ax3, bins=reads_umi_bins)

pp.savefig(fig)
pp.close()


