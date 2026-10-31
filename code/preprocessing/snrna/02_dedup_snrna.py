import gzip
import itertools
from collections import Counter, defaultdict
from pathlib import Path
import pickle

import json
from matplotlib import pyplot as plt
import matplotlib.colors
import numpy as np
import scipy.io
import scipy.sparse
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.neighbors import BallTree
import sys

import multiprocessing as mp
import os
import optparse

sys.path.append("/home/jsilverm/06_synapseseq_repo/synapse_seq_pipeline_code/09_dedup_aav_lib")
import synapse_seq_functions as ssf

#  read in args from cmdline using optparse
# -i input file
# -o output file

parser = optparse.OptionParser()
parser.add_option('-i', '--input', action="store", type="string", default=None)
parser.add_option('-o', '--output', action="store", type="string", default=None)
parser.add_option('-v', '--vtfname', action="store", type="string", default="merged_fuzzyMatch_notMatch_cellID.fastq.gz")
parser.add_option('-b', '--bcfname', action="store", type="string", default="merged_fuzzyMatch_notMatch_cellBC.fastq.gz")
parser.add_option('-u', '--umifname', action="store", type="string", default="merged_fuzzyMatch_notMatch_UMI.fastq.gz")

options, args = parser.parse_args()
print(args)
print(options)

input_dir = options.input
output_dir = options.output
vtfname = options.vtfname
bcfname = options.bcfname
umifname = options.umifname

# if output_dir dir is not present, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("input_dir: ", input_dir)
print("output_dir: ", output_dir)
print("vtfname: ", vtfname)
print("bcfname: ", bcfname)
print("umifname: ", umifname)

sys.stdout.flush()
sys.stderr.flush()

# harcoded paths to whitelists and hamming correction objs
aav_ball_tree_obj_path = "/home/jsilverm/06_synapseseq_repo/synapse_seq_pipeline_code/09_dedup_aav_lib/00_hamming_dist_objs/ball_tree_hamming_2.pkl"
cellranger_cellbarcode_whitelist_path = "/home/jsilverm/06_synapseseq_repo/synapse_seq_pipeline_code/09_dedup_aav_lib/00_hamming_dist_objs/3M-february-2018.txt.gz"


def write_mock_fastqs_and_umi_collapse_cmds_file(cell_name, umis_associated_w_cell, mock_fastq_folder, collapsed_results_folder, umi_collapse_cmd_file=None, log_file_base=None):
    """"
    Writes the fastqs file containing the umis of a given cell, and also writes the umi collapse command needed to be run to dedup that file.
    """
    # create mock fastq file to feed into umicollapse
    mock_fastq_out = os.path.join(mock_fastq_folder, f"{cell_name}.fastq")
    umi_collapse_out = os.path.join(collapsed_results_folder, f"{cell_name}_out.fastq")
    
    
    # create mock fastq file
    quality_string="j"*len(umis_associated_w_cell[0])
    with open(mock_fastq_out, "w") as fh:
        for i, umi in enumerate(umis_associated_w_cell):
            header=f"@umi_{str(i)}"
            fh.write(f"{header}\n{umi}\n+\n{quality_string}\n")

    if os.path.exists(umi_collapse_out):
        os.system(f"rm {umi_collapse_out}")

    cmd = f"umicollapse fastq -k 1 -i {mock_fastq_out} -o {umi_collapse_out} --tag"
    if log_file_base is not None:
        log_file =  os.path.join(log_file_base, f"{cell_name}.log")
        cmd = f"{cmd} > {log_file} 2>&1"
    if umi_collapse_cmd_file is not None:
        with open(umi_collapse_cmd_file, "a") as f:
            f.write(f"{cmd}\n")



def run_umi_collapse_parallel_docker(chunk_inx, cmd_file, cpu_set=[0], mount_dirs=["/home/jsilverm", "/mnt/disks/aca-fastqs/"]):
    """
    Passes in the file with umicollapse commands needed to be run. This then spawns off a docker container to run those commands.
    """
    if chunk_inx % 5 == 0:
        print(f"{chunk_inx}\n")
    group_umi_collapse_execute_file = "/home/jsilverm/06_synapseseq_repo/synapse_seq_pipeline_code/09_dedup_aav_lib/snrna/run_umi_collapse_from_cmd_list.py"
    cpu_set_string="".join(cpu_set)
    
    cmd=f'docker run --cpuset-cpus={cpu_set_string}'
    for inx, dir in enumerate(mount_dirs):
        cmd += f' -v {dir}:{dir}'
    cmd += f' umicollapse_img python {group_umi_collapse_execute_file} -i {cmd_file}'
    os.system(cmd)



def run_snrna_viral_deduplication(input_dir, output_dir, vtfname, bcfname, umifname, aav_ball_tree_obj_path, cellranger_cellbarcode_whitelist_path):

    summary_info_dict = {}

    N_CHUNKS_HEAVY = 100
    N_CORES_HEAVY = 30
    N_CORES_LIGHT = 40
    n_cores_docker_containers=40
    vt_hamming_k = 1

    intermediate_dir = os.path.join(output_dir, "intermediate_files")
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)

    # read in split result from barcode extraction
    print(f"Reading barcodes/identifiers/UMIs")

    sys.stdout.flush()
    sys.stderr.flush()

    cellID_path = os.path.join(input_dir , vtfname)
    cellBC_path = os.path.join(input_dir , bcfname)
    umi_path = os.path.join(input_dir , umifname)

    assert os.path.exists(cellID_path), f"cellID_path does not exist: {cellID_path}"
    assert os.path.exists(cellBC_path), f"cellBC_path does not exist: {cellBC_path}"
    assert os.path.exists(umi_path), f"umi_path does not exist: {umi_path}"



    with gzip.open(cellID_path, "rt") as fh:
        cell_ID = [line.strip() for line in itertools.islice(fh, 0, None, 1)]
    with gzip.open(cellBC_path, "rt") as fh:
        cell_BC = [line.strip() for line in itertools.islice(fh, 0, None, 1)]
    with gzip.open(umi_path, "rt") as fh:
        umis = [line.strip() for line in itertools.islice(fh, 0, None, 1)]

    assert len(cell_ID) == len(cell_BC), "read different number of reads"
    assert len(cell_ID) == len(umis), "read different number of reads"

    total_reads = len(cell_ID)
    print(f"Total of {total_reads} reads")

    summary_info_dict["total_reads_before_dedup"] = total_reads


    ###########################################################
        # Run Hamming Distance Correction for Viral Tags #
    ###########################################################

    # Load the precomputed ball tree object
    with open(aav_ball_tree_obj_path, 'rb') as f:
        ball_tree_obj = pickle.load(f)
    ball_tree = ball_tree_obj["ball_tree"]
    inx_to_vt_seq_dict = ball_tree_obj["inx_to_seq_dict"]

    # Find unique vts present and the indices of reads they came from
    unique_vts = np.unique(cell_ID)
    # create dictionary of VT to a list of indices for where it appears
    vt_to_indices = defaultdict(list)
    for i, vt in enumerate(cell_ID):
        vt_to_indices[vt].append(i)


    summary_info_dict["n_unique_vts_pre_dedup"] = len(unique_vts)
    print(f"Total of {len(unique_vts)} unique VTs")
    print("Running hamming distance correction")
    sys.stdout.flush()
    sys.stderr.flush()

    hamming_correction_results_out_path = os.path.join(intermediate_dir, "hamming_correction_results.pkl")
    if os.path.exists(hamming_correction_results_out_path):
        print(f"Loading hamming correction results from {hamming_correction_results_out_path}")
        hamming_correction_results = pickle.load(open(hamming_correction_results_out_path, "rb"))
    else:
        chunked_small_list = ssf.chunk_seq_list(unique_vts, n_chunks=N_CHUNKS_HEAVY)
        # radius = 3
        pool = mp.Pool(N_CORES_HEAVY)
        args = [(chunk, ball_tree, vt_hamming_k, inx) for inx, chunk in enumerate(chunked_small_list)]
        hamming_correction_results = pool.starmap(ssf.hamming_correct_seq_chunk, args)
        pool.close()
        pool.join()
        print(f"Writing hamming correction results to {hamming_correction_results_out_path}")
        # write results to file
        with open(hamming_correction_results_out_path, "wb") as fh:
            pickle.dump(hamming_correction_results, fh)


    hamming_correction_results_unlisted = list(itertools.chain(*hamming_correction_results))
    sys.stdout.flush()
    sys.stderr.flush()
    
    #Filter criteria:
    # VT is not within specified hamming distance of any other VT. Seen as empty lists of neighbors
    # VT is greater than 0 hamming distance away from more than 1 VT. (ie it is 1 HD away from 2 different VTs)
    parsing_args = [(i, inx_to_vt_seq_dict) for i in hamming_correction_results_unlisted]
    parsed_results = [ssf.parse_VT_hamming_result(*arg) for arg in parsing_args]

    # filter to just indices of valid VTs in the original list
    observed_vt_to_valid_VT_dict = {i["observed_VT"]: i["corrected_VT"] for i in parsed_results if i["is_valid_hamming"]}
    all_valid_vt_indices = list(itertools.chain(*[vt_to_indices[i] for i in observed_vt_to_valid_VT_dict.keys()]))
    sorted_valid_vt_indices = sorted(all_valid_vt_indices)

    ###########################################################
        # Run Hamming Distance Correction for Cell Barcodes #
    ###########################################################
    print("Running hamming distance correction for cell barcodes")
    with gzip.open(cellranger_cellbarcode_whitelist_path, "rt") as fh:
        cb_whitelist = [line.strip() for line in fh]
    cb_whitelist_set = set(cb_whitelist)

    unique_CBs = np.unique(cell_BC)
    # create dictionary of CB to a list of indices for where it appears
    cb_to_indices = defaultdict(list)
    for i, cb in enumerate(cell_BC):
        cb_to_indices[cb].append(i)

    summary_info_dict["n_unique_cbs_pre_dedup"] = len(unique_CBs)
    print(f"Total of {len(unique_CBs)} unique CBs")

    sys.stdout.flush()
    sys.stderr.flush()

    is_valid_unique_CBs = np.array([i in cb_whitelist_set for i in unique_CBs])
    valid_CBs = unique_CBs[is_valid_unique_CBs]
    valid_CBs_present_set = set(valid_CBs)
    invalid_cell_barcodes = unique_CBs[~is_valid_unique_CBs]

    print(f"Total of {len(valid_CBs)} valid CBs")
    print(f"Total of {len(invalid_cell_barcodes)} invalid CBs")

    # Create ball tree of just valid cell barcodes present
    valid_cb_int_encoding = np.array([ssf.convert_to_int_space(i) for i in valid_CBs])
    indices_to_cb_dict = {i: cb for i, cb in enumerate(valid_CBs)}
    ball_tree_cell_barcodes = BallTree(valid_cb_int_encoding, metric="hamming", leaf_size=1)

    # Run hamming correction on invalid cell barcodes
    chunked_seq_list = ssf.chunk_seq_list(invalid_cell_barcodes, n_chunks=N_CHUNKS_HEAVY)
    radius = 1
    args = [(chunk, ball_tree_cell_barcodes, radius, inx) for inx, chunk in enumerate(chunked_seq_list)]
    pool = mp.Pool(N_CORES_HEAVY)
    invalid_cb_result = pool.starmap(ssf.hamming_correct_seq_chunk, args)
    pool.close()
    pool.join()

    invalid_cb_result_unlisted = list(itertools.chain(*invalid_cb_result))

    # parse cell barcode results
    parsed_invalid_cb_results = [ssf.parse_invalid_cb_result(i, indices_to_cb_dict) for i in invalid_cb_result_unlisted]
    observed_cb_to_valid_CB_dict = {i["observed_CB"]: i["corrected_CB"] for i in parsed_invalid_cb_results if i["is_valid_hamming"]}
    for valid_cb in valid_CBs:
        observed_cb_to_valid_CB_dict[valid_cb] = valid_cb

    sorted_valid_indices_cell_bc = sorted(list(itertools.chain(*[cb_to_indices[i] for i in observed_cb_to_valid_CB_dict.keys()])))
    sys.stdout.flush()
    sys.stderr.flush()

    ###########################################################
        # Filter to reads with valid VT and CB #
    ###########################################################

    print("Filtering to reads with valid VT and CB")
    valid_indices = set(sorted_valid_vt_indices).intersection(sorted_valid_indices_cell_bc)
    print(f"n valid indices: {len(valid_indices)}")
    valid_vts = [observed_vt_to_valid_VT_dict[cell_ID[i]] for i in sorted(valid_indices)]
    valid_cbs = [observed_cb_to_valid_CB_dict[cell_BC[i]] for i in sorted(valid_indices)]
    valid_umis = [umis[i] for i in sorted(valid_indices)]

    indices_valid_cb_not_vt = set(sorted_valid_indices_cell_bc).difference(valid_indices)
    indices_valid_vt_not_cb = set(sorted_valid_vt_indices).difference(valid_indices)
    n_valid_cb_not_vt = len(indices_valid_cb_not_vt)
    n_valid_vt_not_cb = len(indices_valid_vt_not_cb)

    summary_info_dict["n_valid_cb_not_vt"] = n_valid_cb_not_vt
    summary_info_dict["n_valid_vt_not_cb"] = n_valid_vt_not_cb
    summary_info_dict["valid_cb_and_vt"] = len(valid_indices)
    summary_info_dict["invalid_cb_and_vt"] = total_reads - len(valid_indices)

    n_unique_cbs_original = len(np.unique(cell_BC))
    n_unique_vts_original = len(np.unique(cell_ID))
    n_unique_cbs_post_dedup = len(np.unique(valid_cbs))
    n_unique_vts_post_dedup = len(np.unique(valid_vts))

    summary_info_dict["n_unique_cbs_original"] = n_unique_cbs_original
    summary_info_dict["n_unique_vts_original"] = n_unique_vts_original
    summary_info_dict["n_unique_cbs_post_dedup"] = n_unique_cbs_post_dedup
    summary_info_dict["n_unique_vts_post_dedup"] = n_unique_vts_post_dedup


    ###########################################################
        # Run Hamming Collapse Graph for UMIs within a corrected cell #
    ###########################################################
    cell_to_umi_vt = defaultdict(list)
    for cell, umi, vt in zip(valid_cbs, valid_umis, valid_vts):
        umi_vt = umi + vt
        cell_to_umi_vt[cell].append(umi_vt)
    
    print(len(cell_to_umi_vt.keys()))
    sys.stdout.flush()
    sys.stderr.flush()
    # umicollapse_folder = os.path.join(intermediate_dir, "umi_collapse_intermediate")
    mock_fastq_folder = os.path.join(intermediate_dir, "mock_fastq")
    collapsed_results_folder = os.path.join(intermediate_dir, "collapsed_results")
    umi_collapse_cmd_file_base = os.path.join(intermediate_dir, "umi_collapse_cmds")
    umi_collapse_log_file_base = os.path.join(intermediate_dir, "umi_collapse_logs")
                                              
    # if not os.path.exists(umicollapse_folder):
    #     os.makedirs(umicollapse_folder)
    if not os.path.exists(mock_fastq_folder):
        os.makedirs(mock_fastq_folder)
    if not os.path.exists(collapsed_results_folder):
        os.makedirs(collapsed_results_folder)
    if not os.path.exists(umi_collapse_cmd_file_base):
        os.makedirs(umi_collapse_cmd_file_base)
    if not os.path.exists(umi_collapse_log_file_base):
        os.makedirs(umi_collapse_log_file_base)
    
    print("Running UMI Collapse for cells")
    sys.stdout.flush()
    sys.stderr.flush()

    # define the number of cores to run docker containers from
    
    # define the number of chunks to break the umi collapse commands into
    n_chunks=500

    # Serially, write the fastq files to be used for dedup, and the commands needed to run them
    umi_collapse_write_args = []
    umi_collapse_command_files = [os.path.join(umi_collapse_cmd_file_base, f"umi_collapse_cmds_{file_inx}") for file_inx in range(n_chunks)]
    for inx, (cell_name, umis_associated_w_cell) in enumerate(cell_to_umi_vt.items()):
        file_inx = inx % n_chunks
        umi_collapse_cmd_file = umi_collapse_command_files[file_inx]
        umi_collapse_write_args.append((cell_name, umis_associated_w_cell, mock_fastq_folder, collapsed_results_folder, umi_collapse_cmd_file, umi_collapse_log_file_base))
    for cmd in umi_collapse_write_args:
        write_mock_fastqs_and_umi_collapse_cmds_file(*cmd)

    # create the args needed to run the collapse commands in seperate docker containers
    docker_umi_collapse_execution_args = []
    for chunk_inx, cmd_file in enumerate(umi_collapse_command_files):
        cpu = chunk_inx % n_cores_docker_containers
        docker_umi_collapse_execution_args.append((chunk_inx, cmd_file, [str(cpu)]))
      
    # run collapses in parallel
    pool = mp.Pool(n_cores_docker_containers)
    pool.starmap(run_umi_collapse_parallel_docker, docker_umi_collapse_execution_args)
    pool.close()
    pool.join()
  
    print("UMI Collapse Done")
    print("Parsing Collapsed Results")
    sys.stdout.flush()
    sys.stderr.flush()

    # parse all umi collapse results in parallel
    cell_names_list = list(cell_to_umi_vt.keys())
    umi_collapse_results_args = [(cell_name, collapsed_results_folder) for cell_name in cell_names_list]
    pool = mp.Pool(N_CORES_LIGHT)
    umi_collapse_results = pool.starmap(ssf.parse_umi_collapse_results, umi_collapse_results_args)
    pool.close()
    pool.join()

    print("Results Parsed, creating output objects")
    sys.stdout.flush()
    sys.stderr.flush()

    # create dictionary with cell_bc as key and umi to corrected umi as value
    cell_bc_to_umi_to_corrected_umi_vt = {cell_name: umi_collapse_results[i] for i, cell_name in enumerate(cell_names_list)}

    umi_split_inx = len(valid_umis[0])

    corrected_umis = []
    for cell_bc, umi, vt in zip(valid_cbs, valid_umis, valid_vts):
        current_umi_vt = umi + vt
        corrected_umi_vt = cell_bc_to_umi_to_corrected_umi_vt[cell_bc][current_umi_vt]
        corrected_umi = corrected_umi_vt[:umi_split_inx]
        corrected_umis.append(corrected_umi)

        
    n_umis_pre_collaspse  = len(np.unique(valid_umis))
    n_umis_post_collaspse = len(np.unique(corrected_umis))
    summary_info_dict["n_umis_pre_collapse"] = n_umis_pre_collaspse
    summary_info_dict["n_umis_post_collapse"] = n_umis_post_collaspse

    ###########################################################
        # Create output objects #
    ###########################################################

    processed_obj = ssf.create_u_and_m(valid_vts, valid_cbs, corrected_umis)
    output_obj_path = os.path.join(output_dir, "dedup_obj.pkl")
    with open(output_obj_path, "wb") as fh:
        pickle.dump(processed_obj, fh)
    # gzip the output obj
    cmd="gzip -f " + output_obj_path
    os.system(cmd)

    # create and save obj of all reads
    all_reads_obj = ssf.create_u_and_m(cell_ID, cell_BC, umis)
    all_reads_obj_path = os.path.join(output_dir, "all_reads_obj.pkl")
    with open(all_reads_obj_path, "wb") as fh:
        pickle.dump(all_reads_obj, fh)
    # gzip the output obj
    cmd="gzip -f " + all_reads_obj_path
    os.system(cmd)

    summary_out_path = os.path.join(output_dir, "dedup_summary_info.json")
    # write summary info to json
    with open(summary_out_path, "w") as fh:
        json.dump(summary_info_dict, fh)


    ###########################################################
        # Create Plots #
    ###########################################################

    out_pdf_name = os.path.join(output_dir, "dedup_summary_plots.pdf")
    pdf = PdfPages(out_pdf_name)

    ### Make Plots ###
    reads_umi_bins = range(0,800, 10)
    fig_reads_umi, ax_reads_umi = plt.subplots() 
    ssf.create_reads_umi_histogram(processed_obj["u_mat"], title_base = "Reads/UMI", bins=reads_umi_bins, ax = ax_reads_umi)
    # add reads/umi plot to pdf
    pdf.savefig(fig_reads_umi)

    read_filters = [0, 10, 50, 100, 200]
    umi_tag_bins = range(0, 20, 1)
    tag_cell_bins = range(0, 20, 1)
    cell_tag_bins = range(0, 20, 1)

    filt_objects = {}

    cb_split_inx = 16
    for read_filter in read_filters:
        filt_obj = ssf.create_u_and_m_mat_from_read_filter(processed_obj["u_mat"], processed_obj["cell_bc_umis"], processed_obj["vts"], threshold_value=read_filter, cb_split_inx=cb_split_inx)
        filt_objects[read_filter] = filt_obj

    for read_filter in read_filters:
        filt_obj = filt_objects[read_filter]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ssf.create_umi_per_tag_histogram(filt_obj["m_mat"], title_base = f"Read Filter {read_filter}", bins=umi_tag_bins, ax = ax1)
        ssf.create_tags_per_cell_histogram(filt_obj["m_mat"], title_base = f"Read Filter {read_filter}", bins=tag_cell_bins, ax = ax2)
        ssf.create_cells_per_tag_histogram(filt_obj["m_mat"], title_base = f"Read Filter {read_filter}", bins=cell_tag_bins, ax = ax3)
        pdf.savefig(fig)
    pdf.close()

    # for each read filter object save, pkl obj
    for read_filter_value in read_filters:
        read_filt_out_folder = os.path.join(output_dir, f"read_filter_{read_filter_value}")
        if not os.path.exists(read_filt_out_folder):
            os.makedirs(read_filt_out_folder)
        
        read_filt_out_path = os.path.join(read_filt_out_folder, "dedup_obj.pkl")
        with open(read_filt_out_path, "wb") as fh:
            pickle.dump(filt_objects[read_filter_value], fh)
        


if __name__ == "__main__":
    run_snrna_viral_deduplication(input_dir, output_dir, vtfname, bcfname, umifname, aav_ball_tree_obj_path, cellranger_cellbarcode_whitelist_path)


