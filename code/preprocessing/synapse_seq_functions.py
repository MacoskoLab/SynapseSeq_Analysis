import itertools
from collections import Counter, defaultdict
from pathlib import Path
import pickle
import matplotlib
import matplotlib.colors
import numpy as np
import scipy.io
import scipy.sparse
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy
import tqdm



def convert_to_int_space(seq):
    nucs = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
    int_space = np.zeros(len(seq))
    for i, nuc in enumerate(seq):
        nuc = nucs.get(nuc)
        int_space[i] = nuc
    int_space = int_space.astype(int)
    return int_space

def chunk_seq_list(seq_list_full, n_chunks=10):
    chunk_size = np.ceil(len(seq_list_full) / n_chunks)
    # create chunked list of tuples of seq and corresponding index in global list
    chunked_result_full = []
    current_chunk = []
    n_tags_total = len(seq_list_full)
    for i in range(n_tags_total):
        current_tup = (i, seq_list_full[i])
        if len(current_chunk) == chunk_size:
            chunked_result_full.append(current_chunk)
            current_chunk = [current_tup]
        else:
            current_chunk.append(current_tup)
    
    # add last chunk
    if len(current_chunk) > 0:
        chunked_result_full.append(current_chunk)

    return chunked_result_full


# return the seq tested as well as all neighbors and distances within hamming radius
def hamming_correct_single_seq(seq_tup, ball_tree, hamming_radius):

    seq_inx = seq_tup[0]
    if seq_inx % 5000 == 0:
        print(f"seq inx: {seq_inx}")
    seq = seq_tup[1]

    seq_len = len(seq)
    seq_int = convert_to_int_space(seq)
    seq_int = seq_int.reshape(1, -1)

    # find all seqs within hamming radius
    model_radius = hamming_radius / seq_len
    neighbors, distances = ball_tree.query_radius(seq_int, r=model_radius, return_distance=True)
    neighbors = neighbors[0]
    distances = distances[0] * seq_len
    return_ele = {"seq": seq, "neighbors": neighbors, "distances": distances, "seq_inx": seq_inx}
    return return_ele


def hamming_correct_seq_chunk(seq_list_chunked, ball_tree, hamming_radius, chunk_inx=None):
    out_string = f"Starting chunk {chunk_inx}"
    print(out_string)
    hamming_result = []
    for seq_tup in seq_list_chunked:
        hamming_result.append(hamming_correct_single_seq(seq_tup, ball_tree, hamming_radius))
    return hamming_result


# return if valid or not, and the final seq to use 
def parse_VT_hamming_result(result, inx_to_seq_dict):
    neighbors = result["neighbors"]
    n_neighbors_in_radius = len(neighbors)
    valid_hamming = True
    # if no neighbors in radius, invalid
    if n_neighbors_in_radius == 0:
        valid_hamming = False
        min_hamming = -1
        closest_seq = None
        min_neighbor_inx = None
    else:
        which_min_inx = np.argmin(result["distances"])
        min_neighbor_inx = neighbors[which_min_inx]
        min_hamming = result["distances"][which_min_inx]
        # invalid of greater than 1 VT is within hamming radius and the min is not an exact match
        if n_neighbors_in_radius > 1 and min_hamming > 0:
            valid_hamming = False
        
        closest_seq = inx_to_seq_dict[min_neighbor_inx]

    res = {
           "is_valid_hamming": valid_hamming, 
           "min_hamming": min_hamming, 
           "n_neighbors_in_radius": n_neighbors_in_radius, 
           "observed_VT": result["seq"], 
           "observed_VT_inx": result["seq_inx"],
           "corrected_VT": closest_seq,
           "corrected_VT_inx_reference_list": min_neighbor_inx
           }
    return res


# if no neighbors in radius, invalid
# if > 1 neighbors in radius, invalid
# if 1 neighbor in radius, valid
def parse_invalid_cb_result(result, inx_to_seq_dict):
    assert "neighbors" in result.keys(), "neighbors not in result"
    neighbors = result["neighbors"]
    n_neighbors_in_radius = len(neighbors)
    valid_hamming = n_neighbors_in_radius == 1
    if valid_hamming:
        min_neighbor_inx = neighbors[0]
        closest_seq = inx_to_seq_dict[min_neighbor_inx]
    else:
        closest_seq = None
        min_neighbor_inx = None
    result = {
        "is_valid_hamming": valid_hamming, 
            "n_neighbors_in_radius": n_neighbors_in_radius,
            "observed_CB": result["seq"],
            "observed_CB_inx": result["seq_inx"],
            "corrected_CB": closest_seq,
            "corrected_CB_inx_reference_list": min_neighbor_inx}

    return result


def create_matrix(data_dict,row_names, col_names):
    m = scipy.sparse.dok_matrix((len(row_names), len(col_names)), dtype=np.int32)
    row2inx = {r: i for i, r in enumerate(row_names)}
    col2inx = {c: i for i, c in enumerate(col_names)}
    for r_name, c_data in data_dict.items():
        for c_name, val in c_data.items():
            row_inx = row2inx[r_name]
            col_inx = col2inx[c_name]
            m[row_inx, col_inx] = val
    m = m.tocsr()
    return m

def create_u_and_m(vts ,cbs ,umis):
    raw_umis_per_cell = defaultdict(lambda: defaultdict(set))
    raw_reads_per_cell_umi = defaultdict(Counter)
    print("Iterating through vts, cbs, and umis")
    for vt, cell_bc, umi in zip(vts, cbs, umis):
        bead_umi_bc = cell_bc + umi

        raw_umis_per_cell[cell_bc][vt].add(umi)
        raw_reads_per_cell_umi[bead_umi_bc][vt] += 1
    

    raw_umis_per_cell = {
        cell_bc: {tag: len(v) for tag, v in tags_per_bead.items()}
        for cell_bc, tags_per_bead in raw_umis_per_cell.items()
    }


    cell_bcs_from_dict = sorted(raw_umis_per_cell)
    vts_from_dict = sorted(set(i for v in raw_umis_per_cell.values() for i in v))
    cell_bcs_umi_from_dict = sorted(raw_reads_per_cell_umi)

    print("Creating m matrix")
    m_mat = create_matrix(raw_umis_per_cell, cell_bcs_from_dict, vts_from_dict)
    print("Creating u matrix")
    u_mat = create_matrix(raw_reads_per_cell_umi, cell_bcs_umi_from_dict, vts_from_dict)

    lib_obj = {"m_mat": m_mat, "u_mat": u_mat, "cell_bcs": cell_bcs_from_dict, "vts": vts_from_dict, "cell_bc_umis": cell_bcs_umi_from_dict}

    return lib_obj

def write_matrix_from_inx_and_lists(data, row_inx, col_inx, row_list, col_list):
    mat = scipy.sparse.dok_matrix((len(row_list), len(col_list)), dtype=np.int32)
    for i, j, k in zip(data, row_inx, col_inx):
        mat[j, k] = i
    mat = mat.tocsr()
    return mat

def readFilt_10xStrat(u_mat):
    cell_argmax=[i[0] for i in u_mat.argmax(axis=1).tolist()]
    
    new_rows=[i for i in range(u_mat.shape[0])]
    new_cols=cell_argmax
    new_data=[u_mat[i,j] for i,j in zip(new_rows,new_cols)]
    
    return new_rows, new_cols, new_data

def filter_to_max_vt_per_cbumi(u_mat, cellbc_umi_ordering, vt_ordering):
    row_inxs, col_inxs, data = readFilt_10xStrat(u_mat)
    u_mat_filt = write_matrix_from_inx_and_lists(data, row_inxs, col_inxs, cellbc_umi_ordering, vt_ordering)
    return u_mat_filt


def u_mat_to_m_mat(u_mat, cellbc_umi_ordering, vt_ordering, cb_split_inx=16):
    # convert cb-umiXvt mat into cbXvt mat
    # assert all rows have only 1 non-zero entry
    n_nonzero_per_row = u_mat.getnnz(axis=1)
    assert all(n_nonzero_per_row == 1), "not all rows have only 1 non-zero entry"

    cb_vt_dict = defaultdict(lambda: defaultdict(int))
    for row_inx, row_name in tqdm.tqdm(enumerate(cellbc_umi_ordering)):
        cb_present = row_name[:cb_split_inx]
        nonzero_inx_at_row = u_mat[row_inx].nonzero()[1][0]
        vt_present = vt_ordering[nonzero_inx_at_row]
        cb_vt_dict[cb_present][vt_present] += 1

    cb_ordering = sorted(cb_vt_dict)
    m_mat = create_matrix(cb_vt_dict, cb_ordering, vt_ordering)
    return m_mat, cb_ordering

def create_u_and_m_mat_from_read_filter(u_mat, cellbc_umi_ordering, vt_ordering, threshold_value, cb_split_inx=16):
    "Do read filter on u_mat, using scritp greater than on threshold value"
    n_nonzero_per_row = u_mat.getnnz(axis=1)
    if not all(n_nonzero_per_row == 1):
        print("Not all rows have only 1 non-zero entry; filtering to max VT per CBUMI")
        u_mat = filter_to_max_vt_per_cbumi(u_mat, cellbc_umi_ordering, vt_ordering)

    max_col_per_row = u_mat.max(axis=1).toarray().flatten()
    is_above_threshold = max_col_per_row >= threshold_value

    new_u_mat = u_mat[is_above_threshold, :]
    u_mat_cellbc_umi_ordering_kept_indices = [i for i, is_above in enumerate(is_above_threshold) if is_above]
    cellbc_umi_ordering = [cellbc_umi_ordering[i] for i in u_mat_cellbc_umi_ordering_kept_indices]

    print("Creating new m_mat")
    m_mat_read_filt, cb_ordering = u_mat_to_m_mat(new_u_mat, cellbc_umi_ordering, vt_ordering, cb_split_inx=cb_split_inx)

    res = {
        "u_mat": new_u_mat,
        "m_mat": m_mat_read_filt,
        "cells": cb_ordering,
        "vts": vt_ordering,
        "cells_umi": cellbc_umi_ordering
    }
    return res


def create_reads_umi_histogram(u_mat, title_base, bins=None, ax=None, truncate_value=None):
    if ax is None:
        fig, ax = plt.subplots()

    # take max of each row
    reads_per_umi = u_mat.max(axis=1).toarray().flatten()
    mean = round(np.mean(reads_per_umi), 3)
    median = round(np.median(reads_per_umi), 3)
    prop_1 = round(np.sum(reads_per_umi == 1) / len(reads_per_umi), 3)
    n = len(reads_per_umi)

    if truncate_value is not None:
        reads_per_umi[reads_per_umi > truncate_value] = truncate_value

    if bins is None:
        bins = 100

    # create fig to return
    title = f"{title_base} Reads per UMI Histogram\nn umi: {n} Mean: {mean}\nMedian: {median}, prop 1: {prop_1}"
    ax.hist(reads_per_umi, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Reads per UMI")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")


def create_umi_per_tag_histogram(m_mat, title_base, bins = None, ax = None):
    fontsize = 10
    if ax is None:
        fig, ax = plt.subplots()

    umis_per_tag = m_mat.data
    mean = round(np.mean(umis_per_tag), 3)
    median = round(np.median(umis_per_tag), 3)
    prop_1 = round(np.sum(umis_per_tag == 1) / len(umis_per_tag), 3)
    n = len(umis_per_tag)

    if bins is None:
        bins = 100

    title = f"{title_base} UMIs per Tag Histogram\nn umis: {n} Mean: {mean}\nMedian: {median}, prop 1: {prop_1}"
    # make title size smaller
    ax.hist(umis_per_tag, bins=bins)
    ax.set_title(title, fontsize=fontsize)

    ax.set_xlabel("UMIs per Tag", fontsize=fontsize)
    ax.set_ylabel("Frequency", fontsize=fontsize)
    ax.set_yscale("log")


def run_umi_collapse_indiv_cell(cell_name, umis_associated_w_cell, mock_fastq_folder, collapsed_results_folder, log_file="/home/jsilverm/logs/umicollapse.log", inx=None):
    if inx is not None:
      if inx % 10 == 1000:
        print(inx)
    # create mock fastq file to feed into umicollapse
    mock_fastq_out = os.path.join(mock_fastq_folder, f"{cell_name}.fastq")
    umi_collapse_out = os.path.join(collapsed_results_folder, f"{cell_name}_out.fastq")

    # if file exists return
    # if os.path.exists(mock_fastq_out) and os.path.exists(umi_collapse_out):
    #     return
    # run umicollapse

    # create mock fastq file
    quality_string="j"*len(umis_associated_w_cell[0])
    with open(mock_fastq_out, "w") as fh:
        for i, umi in enumerate(umis_associated_w_cell):
            header=f"@umi_{str(i)}"
            fh.write(f"{header}\n{umi}\n+\n{quality_string}\n")

    if os.path.exists(umi_collapse_out):
        os.system(f"rm {umi_collapse_out}")
    cmd = f"umicollapse fastq -k 1 -i {mock_fastq_out} -o {umi_collapse_out} --tag > {log_file} 2>&1"
    # cmd = f"docker run --cpus=1 -v /home/jsilverm/:/home/jsilverm umicollapse_img umicollapse fastq -k 1 -i {mock_fastq_out} -o {umi_collapse_out} --tag >> {log_file} 2>&1"
    # cmd = f"docker run --cpuset-cpus=1 -v /home/jsilverm/:/home/jsilverm umicollapse_img umicollapse fastq -k 1 -i {mock_fastq_out} -o {umi_collapse_out} --tag >> {log_file} 2>&1"
    # run the command and wait for it to finish
    os.system(cmd)

def parse_umi_collapse_results(cell_name, umi_collapse_out):
    umi_collapse_out_path = os.path.join(umi_collapse_out, f"{cell_name}_out.fastq")
    assert os.path.exists(umi_collapse_out), f"umi_collapse_out_path does not exist: {umi_collapse_out_path}"


    with open(umi_collapse_out_path) as fh:
        corrected_lines_full = fh.readlines()

    # add all cluster ids to a list
    cluster_id_to_observed_seqs = []
    # create a mapping of cluster id to correct seq
    cluster_id_to_seq_dict = {}
    for i in range(0, len(corrected_lines_full), 4):
        header = corrected_lines_full[i].strip()
        header_split = header.split(" ")
        header_dict = {ele.split("=")[0]: ele.split("=")[1] for ele in header_split[1:]}

        if "cluster_size" in header_dict.keys():
            cluster_size = int(header_dict["cluster_size"])
            cluster_id = header_dict["cluster_id"]
            correct_seq = corrected_lines_full[i+1].strip()
            cluster_id_to_seq_dict[cluster_id] = {"correct_seq": correct_seq, "n_reads": cluster_size}

        observed_seq = corrected_lines_full[i+1].strip()
        header_dict["observed_seq"] = observed_seq
        assert "cluster_id" in header_dict.keys(), f"cluster_id not in cluster_id_to_seq_dict: {header_dict}"
        cluster_id_to_observed_seqs.append(header_dict)


    observed_umi_to_corrected_umi = {}
    for cluster in cluster_id_to_observed_seqs:
        observed_seq = cluster["observed_seq"]
        correct_seq = cluster_id_to_seq_dict[cluster["cluster_id"]]["correct_seq"]
        observed_umi_to_corrected_umi[observed_seq] = correct_seq

    return observed_umi_to_corrected_umi


def create_tags_per_cell_histogram(m_mat, title_base, bins = None, max_tag=None, ax=None, is_slide_seq=False, title_size=12):
    font = 10
    if ax is None:
        fig, ax = plt.subplots()

    if bins is None:
        bins = range(1,10)
    if max_tag is None:
        max_tag = max(bins)
        print(f"max_tag is None, setting to {max_tag}")

    m_mat_binary = (m_mat > 0).astype(int)
    tags_per_cell = np.array(m_mat_binary.sum(axis=1)).flatten()
    tags_per_cell = tags_per_cell.astype(int)

    # tags_per_cell[tags_per_cell > max_tag] = max_tag

    n_cells = len(tags_per_cell)
    mean_tags_per_cell = round(np.mean(tags_per_cell), 3)
    median_tags_per_cell = np.median(tags_per_cell)
    prop_1_tag = round(np.sum(tags_per_cell == 1) / len(tags_per_cell), 3)

    wrapped_title = f"{title_base} Tags per Cell Histogram\nn cells: {n_cells} Mean: {mean_tags_per_cell}\nMedian: {median_tags_per_cell}, prop 1: {prop_1_tag}"
    if is_slide_seq:
        # replace the word cells with beads in title
        wrapped_title = wrapped_title.replace("cell", "bead")

    ax.hist(tags_per_cell, bins=bins)
    ax.set_title(wrapped_title, fontsize=title_size)
    ax.set_xlabel("Tags per Cell")
    # mark each x tick
    # ax.set_xticks(bins)
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    # plt.show()



def create_cells_per_tag_histogram(m_mat, title_base, bins = None, max_cell=None, ax=None, is_slide_seq = False):
    fontsize = 10
    if ax is None:
        fig, ax = plt.subplots()

    if bins is None:
        bins = range(1,10)

    cells_per_tag = np.array(m_mat.sum(axis=0)).flatten()
    cells_per_tag = cells_per_tag.astype(int)

    if max_cell is not None:
      cells_per_tag[cells_per_tag > max_cell] = max_cell
    # remove 0s
    cells_per_tag = cells_per_tag[cells_per_tag != 0]

    n_tags = len(cells_per_tag)
    mean_cells_per_tag = round(np.mean(cells_per_tag), 3)
    median_cells_per_tag = np.median(cells_per_tag)
    prop_1_cell = round(np.sum(cells_per_tag == 1) / len(cells_per_tag), 3)

    wrapped_title = f"{title_base} Cells per Tag Histogram\nn tags: {n_tags} Mean: {mean_cells_per_tag}\nMedian: {median_cells_per_tag}, prop 1: {prop_1_cell}"
    if is_slide_seq:
        # replace the word cells with beads in title
        wrapped_title = wrapped_title.replace("tag", "bead")
    ax.hist(cells_per_tag, bins=bins)
    ax.set_title(wrapped_title, fontsize=fontsize)
    ax.set_xlabel("Cells per Tag", fontsize=fontsize)
    # mark each x tick
    # ax.set_xticks(bins)
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    # plt.show()

def run_row_u_parsing(row, tag_ids, current_row_inx):
    nonzero_indices = row.nonzero()[1]
    nonzero_tags = [tag_ids[i] for i in nonzero_indices]
    nonzero_vals = [row[0,i] for i in nonzero_indices]
    r = {"row_inx": current_row_inx, "nonzero_tags": nonzero_tags, "nonzero_vals": nonzero_vals}
    return r

def run_row_parsing_from_u(u_mat, tag_ids, cell_umi_ids):
    args = [(u_mat.getrow(i), tag_ids, i) for i in range(u_mat.shape[0])]
    n_cores = 20
    pool = mp.Pool(n_cores)
    results = pool.starmap(run_row_u_parsing, args)
    pool.close()

    results_w_cell_name = []
    for i, r in enumerate(results):
        r["umi_cell_name"] = cell_umi_ids[i]
        results_w_cell_name.append(r)

    return results_w_cell_name

def do_hamming_compare_from_max(current_result):
    tags = current_result["nonzero_tags"]
    max_tag_inx = np.argmax(current_result["nonzero_vals"])
    max_tag = tags[max_tag_inx]
    other_tags_non_max = [tags[i] for i in range(len(tags)) if i != max_tag_inx]

    hamming_dists = []
    if len(other_tags_non_max) > 0:
        len_non_max_tag = len(other_tags_non_max[0])
        for i in range(len(other_tags_non_max)):
            hamming_dists.append(scipy.spatial.distance.hamming(list(max_tag), list(other_tags_non_max[i])) * len_non_max_tag)
    hamming_dists = np.array(hamming_dists)
    r = {"row_inx": current_result["row_inx"], "hamming_dists": hamming_dists}
    return r

def do_hamming_compare(current_result):
    len_tags = len(current_result["nonzero_tags"][0])
    tags = current_result["nonzero_tags"]
    hamming_dists = []
    if len(tags) > 1:
        for i in range(len(tags)):
            for j in range(i+1, len(tags)):
                hamming_dists.append(scipy.spatial.distance.hamming(list(tags[i]), list(tags[j])) * len_tags)
    hamming_dists = np.array(hamming_dists)
    r = {"row_inx": current_result["row_inx"], "hamming_dists": hamming_dists}
    return r

def check_hamming_results(cellbc_umi_results, n_cores=20):
    args_tup = [(i,) for i in cellbc_umi_results]
    pool = mp.Pool(n_cores)
    results_hamming = pool.starmap(do_hamming_compare, args_tup)
    pool.close()
    # add all hamming distances to a list
    hamming_dists = []
    for i in results_hamming:
        hamming_dists.extend(i["hamming_dists"])
    return hamming_dists

def check_hamming_results_from_max(cellbc_umi_results, n_cores=20):
    args_tup = [(i,) for i in cellbc_umi_results]
    pool = mp.Pool(n_cores)
    results_hamming = pool.starmap(do_hamming_compare_from_max, args_tup)
    pool.close()
    # add all hamming distances to a list
    hamming_dists = []
    for i in results_hamming:
        hamming_dists.extend(i["hamming_dists"])
    return hamming_dists


def do_basic_hamming(seq1, seq2):
    size = len(seq1)
    hamming = scipy.spatial.distance.hamming(list(seq1), list(seq2)) * size
    return hamming



def get_hamming_null(tags_list, n_comparisons, n_cores=20):
    n_tags = len(tags_list)
    # create n_comparisons where 2 random but not equal numbers between 0 and n_tags are selected
    random_tag_pairs = np.random.choice(n_tags, (n_comparisons, 2), replace=True)

    args = [(tags_list[i], tags_list[j]) for i,j in random_tag_pairs]
    pool = mp.Pool(n_cores)
    results = pool.starmap(do_basic_hamming, args)
    pool.close()
    return results


def parse_raw_umat_for_descrip(u_mat, tag_ids, cell_umi_ids):
    results_w_cell_name = run_row_parsing_from_u(u_mat, tag_ids, cell_umi_ids)
    tags_per_umicell = [len(i["nonzero_tags"]) for i in results_w_cell_name]
    hamming_dists = check_hamming_results(results_w_cell_name, n_cores=20)
    return results_w_cell_name, tags_per_umicell, hamming_dists 


def parse_raw_umat_for_descrip_from_max(u_mat, tag_ids, cell_umi_ids):
    results_w_cell_name = run_row_parsing_from_u(u_mat, tag_ids, cell_umi_ids)
    tags_per_umicell = [len(i["nonzero_tags"]) for i in results_w_cell_name]
    hamming_dists = check_hamming_results_from_max(results_w_cell_name, n_cores=20)
    return results_w_cell_name, tags_per_umicell, hamming_dists