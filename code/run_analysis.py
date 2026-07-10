import os
import sys
import scipy
import pickle
import gzip
import csv 


import numpy as np
import pandas as pd
import anndata as ad

import multiprocessing as mp
import seaborn as sns

import argparse
import statsmodels.api as sm

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from collections import Counter

from scipy.stats import gaussian_kde
from shapely.geometry import LineString
from matplotlib.colors import Normalize


pd.options.mode.chained_assignment = None


class_type_col = "class"
subclass_type_col = "subclass"
cluster_type_col="cluster"
cluster_grouped_col = "cluster_grouped"


dpi = 600

# Filenames located inside the --obj_dir directory
LEAF_TO_SUBCLASS_FILENAME = "leaf_to_subclass.pkl"
PROJECTION_TARGETS_FILENAME = "projection_target_vts.pkl"

# Filenames used only in the aca analysis branch (also inside --obj_dir)
STR_COMBINED_OBJ_FILENAME = "str_combined_obj.pkl"
STR_VT_DF_BEAD_MERGED_FILENAME = "str_vt_df_bead_merged.csv"
MEDULLA_DISSECTION_FILENAME = "medulla_projs_w_umi_support.csv"



def parse_jackknife_results(jackknife_results_lis):

    df_dict = {"cell_type": [], "region": [], "jack_p_val": [], "jack_z_score": [], "n_vts_proj_region": [], "n_vts_ct_total": []}
    n_total_eles = len(jackknife_results_lis)
    for ele_inx, ele in enumerate(jackknife_results_lis):

        p_vals = ele["jackknife_p_vals"]
        z_scores = ele["jackknife_z_scores"]

        jack_p_val = np.mean(p_vals)
        jack_z_score = np.mean(z_scores)
        
        df_dict["cell_type"].append(ele["cell_type"])
        df_dict["region"].append(ele["region"])
        df_dict["jack_p_val"].append(jack_p_val)
        df_dict["jack_z_score"].append(jack_z_score)
        df_dict["n_vts_proj_region"].append(ele["n_vts_proj_region"])
        df_dict["n_vts_ct_total"].append(ele["n_vts_ct_total"])

    df_res = pd.DataFrame(df_dict)
    df_res["prop_vts"] = df_res["n_vts_proj_region"] / df_res["n_vts_ct_total"]
    return df_res

def bootstrap_null_weighted_mean(null_scores, n_vts_compare, NBOOT=1000):
    weighted_means = []
    # return cdf_ct
    for inx in range(NBOOT):
        sampling = np.random.choice(null_scores, n_vts_compare, replace=True)
        weighted_mean = np.mean(sampling)
        weighted_means.append(weighted_mean)
    return np.array(weighted_means)

def preform_weighted_vt_mean_region_w_jackknife(jackknife_args):
    cell_type_scores = jackknife_args["cell_type_scores"]
    null_scores = jackknife_args["null_scores"]
    cell_type = jackknife_args["cell_type"]
    region_vts = jackknife_args["region_vts"]
    region_name = jackknife_args["region_name"]
    jackknife_holdout_prop = jackknife_args.get("jackknife_holdout_prop", None)
    jackknife_n_holdout = jackknife_args.get("jackknife_n_holdout", None)
    n_jackknives = jackknife_args.get("n_jackknives", 1000)
    NBOOT = jackknife_args.get("NBOOT", 1000)
    inx_outer = jackknife_args.get("inx", None)
    seed = jackknife_args.get("seed", None)
    if seed is not None:
        np.random.seed(seed)
        

    if (jackknife_holdout_prop is None)  and (jackknife_n_holdout is None):
        raise Exception("Must provide proprtion or N to holdout")

    if (jackknife_holdout_prop is not None)  and (jackknife_n_holdout is not None):
        raise Exception("Both n and proportion provided. Must leave one None")
    

    n_vts_ct_total = len(cell_type_scores)
    n_vts_proj_region = np.sum(cell_type_scores > 0)

    if jackknife_holdout_prop is not None:
        ct_jackknifing_prop = 1-jackknife_holdout_prop
        n_vts_jackknifing = int(np.ceil(len(cell_type_scores) * ct_jackknifing_prop))
    elif jackknife_n_holdout is not None:
        n_vts_jackknifing = len(cell_type_scores) - jackknife_n_holdout

    if (jackknife_n_holdout is None) or jackknife_n_holdout > 1:
        jackknifed_means = []
        for _ in range(n_jackknives):
            cell_type_scores_sampled = np.random.choice(cell_type_scores, n_vts_jackknifing, replace=True)
            c_mean = np.mean(cell_type_scores_sampled)
            jackknifed_means.append(c_mean)
    else:
        # preforming hold out each single vt
        n_jackknives = len(cell_type_scores)
        jackknifed_means = []
        for inx in range(n_jackknives):
            cell_type_scores_one_removed = np.delete(cell_type_scores, inx)
            c_mean = np.mean(cell_type_scores_one_removed)
            jackknifed_means.append(c_mean)
    
    null_weighted_means = bootstrap_null_weighted_mean(null_scores,n_vts_jackknifing, NBOOT=NBOOT)
    std_null_weighted_means = np.std(null_weighted_means)
    mean_null_weighted_means = np.mean(null_weighted_means)
    
    if std_null_weighted_means == 0:
        std_null_weighted_means = 0.00001
        
    jackknife_p_vals = []
    jackknife_z_scores = []
    for inx in range(n_jackknives):
        c_mean = jackknifed_means[inx]
        p_val = np.mean(null_weighted_means >= c_mean)
        z_score = (c_mean - mean_null_weighted_means) / std_null_weighted_means
        jackknife_p_vals.append(p_val)
        jackknife_z_scores.append(z_score)

    zscore_mean = np.mean(jackknife_z_scores)
    
    res = {"cell_type": cell_type, 
           "region": region_name,
           "jackknife_p_vals": jackknife_p_vals,
           "jackknife_z_scores": jackknife_z_scores,
           "zscore_mean": zscore_mean,
           "std_null_weighted_means": std_null_weighted_means,
           "mean_null_weighted_means": mean_null_weighted_means,
           "n_vts_proj_region": n_vts_proj_region,
           "n_vts_ct_total": n_vts_ct_total,
           "jackknifed_means": jackknifed_means,
           "null_weighted_means": null_weighted_means
          }
    return res


def preform_all_celltype_region_comparisons_jackknife(cell_vt_df, proj_dict, score_col, cell_type_col = subclass_type_col, null_type="Glia", jackknife_holdout_prop=None, jackknife_n_holdout=None, n_jackknives = 10000, min_ct_n = 100, NBOOT=1000, return_unparsed=False, subset_mscar=True, seed=0, do_parallel=True):
    np.random.seed(seed)
    
    if (jackknife_holdout_prop is None)  and (jackknife_n_holdout is None):
        raise Exception("Must provide proprtion or N to holdout")

    if (jackknife_holdout_prop is not None)  and (jackknife_n_holdout is not None):
        raise Exception("Both n and proportion provided. Must leave one None")
    if subset_mscar:
        cell_vt_df = cell_vt_df[cell_vt_df["is_mscar"]]
    
    unique_regions = list(proj_dict.keys())

    cell_type_counts = cell_vt_df[cell_type_col].value_counts()
    valid_cts = cell_type_counts[cell_type_counts >= min_ct_n].index

    args = []
    arg_inx = 0
    total_regions = len(unique_regions)
    for region_inx, region in enumerate(unique_regions):
        
        region_vts = proj_dict[region]
        cell_vt_df_region = cell_vt_df.copy()
        projects_region = cell_vt_df_region["vt"].isin(region_vts)
        cell_vt_df_region["region_score"] = cell_vt_df_region[score_col]
        cell_vt_df_region.loc[~projects_region, "region_score"] = 0
        
        cell_type_to_vt_scores = cell_vt_df_region.groupby(cell_type_col)["region_score"].apply(list).to_dict()
        null_scores = cell_type_to_vt_scores.pop(null_type)

        for inx, cell_type in enumerate(valid_cts):
            if cell_type == null_type:
                continue
            cell_type_scores = np.array(cell_type_to_vt_scores[cell_type])
            jackargs_dict = {"cell_type_scores":cell_type_scores,
                            "null_scores":null_scores,
                             "cell_type": cell_type,
                             "region_vts": region_vts,
                             "region_name": region,
                             "jackknife_holdout_prop": jackknife_holdout_prop,
                             "n_jackknives":n_jackknives,
                             "jackknife_n_holdout": jackknife_n_holdout,
                             "inx": arg_inx,
                             "NBOOT": NBOOT
                            }
            c_args = (jackargs_dict,)
            args.append(c_args)
            arg_inx+=1

    if do_parallel:
        n_cores = 30
        pool = mp.Pool(n_cores)
        results = pool.starmap(preform_weighted_vt_mean_region_w_jackknife, args)
        pool.close()
        pool.join()
    else:
        results = [preform_weighted_vt_mean_region_w_jackknife(*arg) for arg in args]

    if return_unparsed:
        return results

    jack_df = parse_jackknife_results(results)

    return jack_df


def create_type_to_region_prop_cell_match(cell_vt_df, proj_regions, cell_type_col, return_prop=True):

    region_type_matching_rate = {}
    for region_name, vts in proj_regions.items():
        print(region_name)
        print(len(vts))
        cell_vt_df["matches_region"] = cell_vt_df["vt"].isin(vts)
        cellular_matching_df = cell_vt_df.groupby("cell").agg({cell_type_col: "first", "matches_region": "any"})
        if return_prop:
            type_match_rate_df = cellular_matching_df.groupby(cell_type_col)["matches_region"].mean().reset_index()
        else:
            type_match_rate_df = cellular_matching_df.groupby(cell_type_col)["matches_region"].sum().reset_index()

        type_match_rate_df["region"] = region_name
        type_match_rate_df.index = type_match_rate_df[cell_type_col]

        type_match_dict = type_match_rate_df["matches_region"].to_dict()
        
        region_type_matching_rate[region_name] = type_match_dict
    return region_type_matching_rate


def create_basic_dotplot_proj_sub_groupings_continous(proj_df, cell_type_col, region_col, value_col, color_col, text_col,
                                 x_axis_grouping_tup_list = None, last_ordering=["empty"], title="CT Projections",
                                 cell_type_ordering=None, region_ordering=None ,
                                 show_intersections_by_ct = True, color_scale_min=-3, color_scale_max=None, 
                                sig_val=0.05, min_show_val=0.2, size_scale=1, alt_size=10, negate_sig_val=False,
                                 only_show_text_if_sig=False, pdf=None, only_show_projecting_cts=True, show_text=True, 
                                                      vmax=0.1, vmin=0, xtick_fsize=12, ytick_fsize=12,min_n_red_circle=0,
                                                      save_path=None, space_per_ct=1, space_per_region=1, show_p_val_legend=True,
                                                      keep_types=None, sizing_factor_x=0, sizing_factor_y=0, cell_match_rates_dict=None,
                                                      sig_circle_width=1.5, total_intersects_by_ct={}
                                                     ):

    # Set sizing function for dots on the plot
    # sizing_func = lambda x: 1000 * np.log10((1/x))
    # sizing_func = lambda x: x * 400 if x > 2 else 10

    def sizing_func(p_val, show_p_val=0.35, max_p=0, scale=size_scale, min_size=alt_size, alt_size=alt_size):
        ref_p_sizes = {0: 1000, 0.001: 500, 0.01: 250, 0.1: 125, 0.2: 30, 0.3: 15, 0.5: 5}
        ref_p_sizes_scaled = {p: size*scale for p, size in ref_p_sizes.items()}
        max_size = ref_p_sizes_scaled[0]
        if p_val >= show_p_val:
            # return min_size
            return 0
        
        sorted_p_vals = sorted(ref_p_sizes_scaled.keys())
        
        for i in range(len(sorted_p_vals) - 1):
            lower_p = sorted_p_vals[i]
            upper_p = sorted_p_vals[i + 1]
            
            if lower_p <= p_val < upper_p:
                lower_size = ref_p_sizes_scaled[lower_p]
                upper_size = ref_p_sizes_scaled[upper_p]
                
                # Linear interpolation
                slope = (upper_size - lower_size) / (upper_p - lower_p)
                interpolated_size = lower_size + slope * (p_val - lower_p)
                
                return max(min_size, min(max_size, interpolated_size))
        
        # If p_val is smaller than the smallest reference p-value
        return max_size

    def color_func(prop_val, vmax=0.1, vmin=0):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        normalized_val = norm(prop_val)
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["#d0d0d0", "#1e3a8a"])
        rgb_color = cmap(normalized_val)
        hex_color = mcolors.rgb2hex(rgb_color)
        
        return hex_color

    n_cts = proj_df[cell_type_col].nunique()
    n_regions = proj_df[region_col].nunique()
    width = n_cts * space_per_ct
    height = n_regions * space_per_region
    plt.figure(dpi=dpi)
    fig, ax = plt.subplots(figsize=(width, height))
    unique_col_present  = proj_df[cell_type_col].unique()

    #### Setting cell type grouping info ####
    if x_axis_grouping_tup_list is not None:
        
        order_x_alphabetically=False

        x_labels_list = []
        break_points = []
        current_x_inx = 0

        all_leaves_in_group = set([ele for tup in x_axis_grouping_tup_list for ele in tup[1]])
        unique_cols_not_in_group = [ele for ele in unique_col_present if ele not in all_leaves_in_group]
        unknown_tup = ("unknown", unique_cols_not_in_group)
        # add to front
        x_axis_grouping_tup_list = [unknown_tup] + x_axis_grouping_tup_list
        # loop through the groups and add the leafs in that group to the xlabels if they are present
        for group, leaf_list in x_axis_grouping_tup_list:
            current_group_leaf_list = []
            for leaf_name in leaf_list:
    
                if leaf_name not in unique_col_present:
                    continue

                current_group_leaf_list.append(leaf_name)
            if len(current_group_leaf_list) == 0:
                continue

            x_labels_list.extend(current_group_leaf_list)
            current_break_point_ele = {"name": group, "start": current_x_inx}
            current_x_inx += len(current_group_leaf_list)
            current_break_point_ele["end"] = current_x_inx
            break_points.append(current_break_point_ele)
        unique_cell_types = np.array(x_labels_list)
    else:
        if cell_type_ordering is None:
            # reorder xlabels such that glia and empties are last
            xlabels = [ele for ele in unique_col_present if ele not in last_ordering]
            last_ordering_present_in_data = [ele for ele in last_ordering if ele in unique_col_present]
            xlabels.extend(last_ordering_present_in_data)
            unique_cell_types = np.array(xlabels)
        else:
            unique_cell_types = np.array(cell_type_ordering)              


    if region_ordering is None:
        unique_regions = proj_df[region_col].unique()
        unique_regions = np.sort(unique_regions)
    else:
        unique_regions = np.array(region_ordering)
    n_regions = len(unique_regions)
    #### Loop for placing dots ####
    for index, row in proj_df.iterrows():
        # Get the x and y values
        cell_type = row[cell_type_col]
        region = row[region_col]

        if cell_type not in unique_cell_types:
            continue

        row_inx = np.where(unique_regions == region)[0][0]
        col_inx = np.where(unique_cell_types == cell_type)[0][0]

        # Determine if the point is significant
        current_value = round(row[value_col], 4)
        n_projs_region = row["n_vts_proj_region"]
        if negate_sig_val:
            is_sig = (current_value <= sig_val) & (n_projs_region >= min_n_red_circle)
        else:
            is_sig = (current_value >= sig_val) & (n_projs_region >= min_n_red_circle)
        pt_size = sizing_func(current_value)

        # Plot the point
        if cell_match_rates_dict is not None:
            color = cell_match_rates_dict[region][cell_type]
        else:
            color=row[color_col]
        current_color_val = color_func(color, vmax=vmax, vmin=vmin)

        ax.scatter(col_inx, row_inx, c=current_color_val, s = pt_size, cmap="Blues", vmin=color_scale_min, vmax=color_scale_max)

        # Add a red circle if the point is significant
        if is_sig:
            ax.scatter(col_inx, row_inx, s=pt_size, facecolors='none', edgecolors='red', linewidths=sig_circle_width)
        text_val = round(row[text_col], 3)
        if (show_text and (is_sig or (not only_show_text_if_sig))):
            ax.text(col_inx, row_inx, f"{text_val}", ha='center', va='center', weight="bold")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)    
    # Create a color map (blues)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#d0d0d0", "#1e3a8a"])
    # Create a ScalarMappable object
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # This line is necessary for the colorbar to work correctly

    # Add the colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.tick_params(labelsize=14)                           
            
    ymin, ymax = ax.get_ylim()
    # Display sum_intersection_by_ct values at the top of each column
    if show_intersections_by_ct:

        for col_inx, cell_type in enumerate(unique_cell_types):
            # if cell_type in total_intersects_by_ct:
            value = total_intersects_by_ct.get(cell_type, 0)
                
            ax.text(col_inx, n_regions-0.3, f"{value}", ha='center', va='top', rotation=90, weight="bold")

    # Add lines denoting the groupings of differnt sub cell types
    if x_axis_grouping_tup_list is not None:
        for break_point in break_points:
            ax.axvline(break_point["end"] - 0.5, color="black")
            if show_text:
                mid_point = ((break_point["start"] + break_point["end"]) / 2) - 1
                # move cell type grouping text down 50 units
                ax.text(mid_point, ymax, break_point["name"], rotation=90, weight="bold")

    # Set the labels for the x and y axes
    # ax.set_xlabel(cell_type_col)
    # ax.set_ylabel(region_col)
    ax.set_xticks(np.arange(len(unique_cell_types)))
    ax.set_xticklabels(unique_cell_types, rotation=90, fontsize=xtick_fsize)
    ax.set_yticks(np.arange(len(unique_regions)))
    ax.set_yticklabels(unique_regions, fontsize=ytick_fsize)
    ax.grid(False)
    # title_y=1

    if title == "":
        title = "P-value"

    # Get the current limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Expand the limits by a certain factor (e.g., 10%)
    # expand_factor = -0.04
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    ax.set_xlim(x_min - sizing_factor_x * x_range, x_max + sizing_factor_x * x_range)
    ax.set_ylim(y_min - sizing_factor_y * y_range, y_max + sizing_factor_y * y_range)

    # if show_intersections_by_ct:
    title_y = 1.35
                                                         
    p_vals_legend_pos = 1.3

    if show_p_val_legend:
        size_legend = [0, 0.001, 0.01, 0.05, 0.1, 0.2]
        size_legend_sizes = [sizing_func(i) for i in size_legend]
        size_legend_labels = [f'{i}' for i in size_legend]
        handles = [plt.scatter([], [], color='grey', s=size) for size in size_legend_sizes]
        size_legend = ax.legend(handles, size_legend_labels, title='', loc="upper center", 
                                ncol=len(size_legend), handletextpad=1.5, bbox_to_anchor=(0.5, p_vals_legend_pos))
        size_legend.set_frame_on(False)

    ax.set_title(title, y=title_y, fontdict={"fontsize":12})
                                                         
    # Adjust the layout to make room for the legend
    plt.tight_layout()
    # plt.subplots_adjust(top
    if save_path is not None:
        fig.savefig(save_path, transparent=True)

    return fig


def plot_patch_projections_figure(cell_type_results, cell_type_name, combined_obj, cgrid_str_spline_subset, polygon_border, combined_bead_coord_df, coords_df_region_col, bin_key="int_bin", results_region_col="bead_name", title=None, show_p_val=True,sig_val=0.05, ax=None):
    # cell_type_results = spline_results.loc[spline_results["cell_type"] == cell_type_name]
    if title is None:
        title = cell_type_name
    # create curves defining patch boundaries, use str_polygon and the surrounding patches as borders
    if ax is None:
        fig, ax = plt.subplots()


    result_regions = cell_type_results[results_region_col].unique()
    region_to_p_val_dict = {region: p_val for region, p_val in zip(cell_type_results[results_region_col], cell_type_results["jack_p_val"])}

    combined_bead_coord_df.index = combined_bead_coord_df["region_bead"]
    combined_bead_coord_df["region_p_val"] = combined_bead_coord_df[bin_key].apply(lambda x: region_to_p_val_dict[x])
    bead_to_p_val = combined_bead_coord_df["region_p_val"].to_dict()
    sig_regions = [region_name for region_name, p_val in region_to_p_val_dict.items() if p_val <= sig_val]
    sig_beads = set(combined_bead_coord_df.loc[combined_bead_coord_df[coords_df_region_col].isin(sig_regions), "region_bead"].values)

    plot_str_from_combined_obj_sig_beads(combined_obj, ax=ax, polygon_interest = polygon_border, bead_to_p_val_dict=bead_to_p_val, sig_val=0.05)

    grouped = cgrid_str_spline_subset.groupby('r_inx_reset')

    # # Create a new figure
    # Iterate through each R group and plot a line
    n_groups_total = len(grouped)
    for iter_inx, (r_inx, group) in enumerate(grouped):
        # place p value at top of line
        # removed clause, as cgrid_str no longer has lowest bound

        if True:
            line = LineString(zip(group['x'], group['y']))
            # Clip the line with the polygon
            clipped_line = line.intersection(polygon_border)
            ax.plot(*clipped_line.xy, color='black', linewidth=1)
    
        if r_inx not in region_to_p_val_dict:
            continue
        if show_p_val:
            p_val = round(region_to_p_val_dict[r_inx], 3)
            is_sig = p_val < sig_val
            if is_sig:
                p_val_text = f"{p_val}*"
            else:
                p_val_text = str(p_val)
                # Find the maximum y-value to place the text above the line
            max_y = group['y'].max()
            max_x = group['x'][group['y'].idxmax()]

        #     # Add the p-value text above the line
            ax.annotate(f'{p_val_text}', 
                        xy=(max_x, max_y),
                        fontsize=10,
                        xytext=(-12, 9),  # 5 points vertical offset
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        weight='bold' if is_sig else 'normal')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(*polygon_border.exterior.xy, color='black', linestyle=':', linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return ax

def plot_subpatch_results(patch_results, cell_type_name, combined_obj, cgrid_str_spline_subset, polygon_border, combined_bead_coord_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    is_type = patch_results["cell_type"] == cell_type_name
    ct_results_spec = patch_results[is_type]
    plot_patch_projections_figure(ct_results_spec, cell_type_name, combined_obj=combined_obj, cgrid_str_spline_subset=cgrid_str_spline_subset, polygon_border=polygon_border, combined_bead_coord_df=combined_bead_coord_df, coords_df_region_col="int_bin",  results_region_col ="region", title="", show_p_val=False, ax=ax)


def add_glia_inh_column(cell_vt_df, col_update, col_has_glia="subclass_visual"):

    new_col_name = f"{col_update}_inh_glia"
    cell_vt_df[new_col_name] = cell_vt_df[col_update]
    is_glia = cell_vt_df[col_has_glia] == "Glia"
    is_inh = cell_vt_df[class_type_col] == "Inh"
    is_glia_inh = is_glia | is_inh
    cell_vt_df.loc[is_glia_inh, new_col_name] = "Inh-Glia"

def subset_to_valid_subclasses(vt_df, valid_subclasses, subclass_col="subclass_visual"):
    is_valid_subclass = vt_df[subclass_col].isin(valid_subclasses)
    vt_df_valid_sub = vt_df.loc[is_valid_subclass]
    return vt_df_valid_sub



def get_region_and_max_prop(group, region_col):

    n_umi_vec = group["n_umi"]
    region_vec = group[region_col]
    max_inx = n_umi_vec.idxmax()
    region_name = region_vec.loc[max_inx]
    max_umi_val = n_umi_vec.loc[max_inx]
    total_umis = n_umi_vec.sum()
    prop_max = max_umi_val / total_umis

    res = {"region": region_name, "prop_max": prop_max, "max_umi_count": max_umi_val}
    return res

def parse_max_results(region_umi_prop_dict):
    res_data_dict = {"vt": [], "region": [], "prop_max": [], "max_umi_count": []}
    for vt, ele in region_umi_prop_dict.items():
        res_data_dict["vt"].append(vt)
        res_data_dict["region"].append(ele["region"])
        res_data_dict["prop_max"].append(ele["prop_max"])
        res_data_dict["max_umi_count"].append(ele["max_umi_count"])
    df = pd.DataFrame(res_data_dict)
    return df
    
def compute_max_umi_region_df(med_vt_umi_count_df, et_med_proj_matching_df_mscar, region_col_med="y", min_prop_region = 0.8):
    
    med_vt_umi_med_region_df = med_vt_umi_count_df.groupby(["vt", region_col_med])["n_umi"].sum().reset_index()
    region_umi_prop_med_regions = med_vt_umi_med_region_df.groupby("vt").apply(lambda group: get_region_and_max_prop(group, region_col_med)).to_dict()
    region_umi_med_regions_df = parse_max_results(region_umi_prop_med_regions)
    
    n_vts  = len(region_umi_med_regions_df["vt"].unique())
    
    is_valid_vt_prop = region_umi_med_regions_df["prop_max"] >= min_prop_region
    has_valid_umi_count = region_umi_med_regions_df["max_umi_count"] > 1
    
    is_valid_vt = is_valid_vt_prop & has_valid_umi_count
    region_umi_df_valid = region_umi_med_regions_df[is_valid_vt].copy()
    new_col=f"{region_col_med}_placed"
    region_umi_df_valid.index = region_umi_df_valid["vt"]
    region_umi_df_valid[new_col] = region_umi_df_valid["region"]
    vt_to_region_dict = region_umi_df_valid["region"].to_dict()

    et_med_proj_matching_df_mscar_region_max_umi = et_med_proj_matching_df_mscar.copy()
    is_valid_vt_mask = et_med_proj_matching_df_mscar_region_max_umi["vt"].isin(vt_to_region_dict)
    et_med_proj_matching_df_mscar_region_max_umi = et_med_proj_matching_df_mscar_region_max_umi[is_valid_vt_mask].copy()

    et_med_proj_matching_df_mscar_region_max_umi[new_col] = et_med_proj_matching_df_mscar_region_max_umi["vt"].apply(lambda vt:vt_to_region_dict[vt])
    # et_med_proj_matching_df_mscar_region_max_umi = et_med_proj_matching_df_mscar_region_max_umi.groupby('vt').first().reset_index()
    return et_med_proj_matching_df_mscar_region_max_umi, region_umi_df_valid
    
def compute_mean_and_conf_intervals(ct_mean_results):

    mean_conf_intervals = {ct_name: get_mean_95_conf(c_mean) for ct_name, c_mean in ct_mean_results.items()}
    res_df_dict = {
        "ct_name": [],
        "mean": [],
        "lb": [],
        "ub": []
    }
    for ct_name, mean_obj in mean_conf_intervals.items():
        mean = mean_obj["mean"]
        lb = mean_obj["lb"]
        ub = mean_obj["ub"]
    
        res_df_dict["ct_name"].append(ct_name)
        res_df_dict["mean"].append(mean)
        res_df_dict["lb"].append(lb)
        res_df_dict["ub"].append(ub)
    
    mean_95_conf_df = pd.DataFrame(res_df_dict)

    mean_95_conf_df["upper_error"] = mean_95_conf_df["ub"] - mean_95_conf_df["mean"]
    mean_95_conf_df["lower_error"] = mean_95_conf_df["mean"] - mean_95_conf_df["lb"]
    mean_95_conf_df["group_display"] = mean_95_conf_df["ct_name"].apply(lambda x: x[0])

    
    return mean_95_conf_df

def plot_mean_error_bars(mean_95_conf_df_non_null, ax=None, title="", order=None, group_col="ct_name", lb_name="lower_error", ub_name="upper_error"):
    if ax is None:
        fig, ax = plt.subplots()

    # If order is provided, reorder the dataframe
    if order is not None:
        mean_95_conf_df_non_null = mean_95_conf_df_non_null.set_index('ct_name').loc[order].reset_index()

    ax.errorbar(
        x = mean_95_conf_df_non_null[group_col], 
        y = mean_95_conf_df_non_null["mean"],
        yerr=[mean_95_conf_df_non_null[lb_name], mean_95_conf_df_non_null[ub_name]], 
        fmt='o', capsize=5, capthick=2, ecolor='black',  color='black',
    )

    # Set the x-axis tick labels in the specified order
    if order is not None:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
    
    ax.set_title(title)

    # Rotate x-axis labels for better readability if needed
    plt.xticks(rotation=45, ha='right')
    ax.grid(False)
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

def get_mean_95_conf(lis):
    sorted_lis = sorted(lis)
    percentiles = np.percentile(sorted_lis, [2.5, 97.5])
    lb = percentiles[0]
    ub = percentiles[1]
    mean = round(np.mean(lis), 3)

    res = {
        "mean": mean,
        "lb": lb,
        "ub": ub
    }
    return res
    
def plot_group_conf_interval(vt_matching_res, dissectate_grouping, coord_interest, N_BOOT=1000, score_col="prob_attr", title = "", ax = None, order=None, x_gradient=[], y_gradient=[]):
    # print(vt_matching_res[[dissectate_grouping, coord_interest, score_col]])
    group_mean_results = {}
    unique_groups = vt_matching_res[dissectate_grouping].unique()
    for group in unique_groups:
        current_group_means = []
        group_data = vt_matching_res.loc[vt_matching_res[dissectate_grouping] == group, ]
        if group_data.shape[0] < 1:
            continue
        for iter in range(N_BOOT):
            n_points = group_data.shape[0]
            sampled_indices = np.random.choice(n_points, n_points, replace=True)
            sampled_df = group_data.iloc[sampled_indices]
            c_mean = weighted_mean(sampled_df, coord_interest, score_col)
            current_group_means.append(c_mean)
        group_mean_results[group] = current_group_means
    

    conf_interval_res = compute_mean_and_conf_intervals(group_mean_results)
    
    conf_interval_res = conf_interval_res.sort_values("ct_name")

    error_size =  np.max(conf_interval_res["mean"]) - np.min(conf_interval_res["mean"])

    has_neg_lower_error = conf_interval_res["lower_error"] <= 0
    has_neg_upper_error =  conf_interval_res["upper_error"] <= 0

    has_poor_error = has_neg_lower_error | has_neg_upper_error
    conf_interval_res.loc[has_poor_error, ["lower_error", "upper_error"]] = error_size
    plot_mean_error_bars(conf_interval_res, title=title, ax=ax, order=order)

    return conf_interval_res

def weighted_mean(df, value_col, weight_col):
    return (df[value_col] * df[weight_col]).sum() / df[weight_col].sum()


def add_dissectate_grouping_vars_cell_vt_df(cell_vt_df, cell_col="cell"):
    dissectate_grouping_dict = {
        "A": "A-B",
        "B": "A-B",
        "C": "C-D",
        "D": "C-D",
        "E": "E-F",
        "F": "E-F",
        "G": "G-H",
        "H": "G-H",
        "I": "I-J",
        "J": "I-J",
        "K": "K-L",
        "L": "K-L"
    }
    
    dissectate_grouping_dict_3 = {
        "A": "A-B-C",
        "B": "A-B-C",
        "C": "A-B-C",
        "D": "D-E-F",
        "E": "D-E-F",
        "F": "D-E-F",
        "G": "G-H-I",
        "H": "G-H-I",
        "I": "G-H-I",
        "J": "J-K-L",
        "K": "J-K-L",
        "L": "J-K-L"
    }
    
    dissectate_grouping_dict_4 = {
        "A": "A-B-C-D",
        "B": "A-B-C-D",
        "C": "A-B-C-D",
        "D": "A-B-C-D",
        "E": "E-F-G-H",
        "F": "E-F-G-H",
        "G": "E-F-G-H",
        "H": "E-F-G-H",
        "I": "I-J-K-L",
        "J": "I-J-K-L",
        "K": "I-J-K-L",
        "L": "I-J-K-L"
    }
    
    
    dissectate_grouping_dict_medial_lateral = {
        "A": "Medial",
        "B": "Medial",
        "C": "Medial",
        "D": "Medial",
        "E": "Medial",
        "F": "Medial",
        "G": "Lateral",
        "H": "Lateral",
        "I": "Lateral",
        "J": "Lateral",
        "K": "Lateral",
        "L": "Lateral"
    }
    
    cell_vt_df["dissectate"] = cell_vt_df[cell_col].apply(lambda x: x[0])
    cell_vt_df["medial_lateral_grouping"] =  cell_vt_df["dissectate"].apply(lambda x: dissectate_grouping_dict_medial_lateral[x])
    cell_vt_df["dissectate_grouped_2"] =  cell_vt_df["dissectate"].apply(lambda x: dissectate_grouping_dict[x])
    cell_vt_df["dissectate_grouped_3"] =  cell_vt_df["dissectate"].apply(lambda x: dissectate_grouping_dict_3[x])
    cell_vt_df["dissectate_grouped_4"] =  cell_vt_df["dissectate"].apply(lambda x: dissectate_grouping_dict_4[x])



def weighted_density_contour(vt_matching_res, cell_type_col, cell_type_name, weight_col, polygon=None, bins=100, levels=10, ax=None, combined_obj=None, color=None, return_quant_res=False, title=None, show_color_bar=True):

    if title is None:
        # title = cell_type_name
        title = ""
    cell_type_df = vt_matching_res.loc[vt_matching_res[cell_type_col] == cell_type_name]

    # Create a path from the polygon for efficient point-in-polygon testing
    polygon_path = Path(np.array(polygon.exterior.coords))
    
    # Filter points to only those inside the polygon
    points_inside = [Point(row['x_coord'], row['y_coord']) for _, row in cell_type_df.iterrows()]
    mask = polygon.contains(points_inside)
    df_inside = cell_type_df[mask]

    # Extract data from filtered dataframe
    x = df_inside["x_coord"]
    y = df_inside["y_coord"]
    z = df_inside[weight_col]
    
    # Create mesh grid
    x_range = polygon.bounds[2] - polygon.bounds[0]
    y_range = polygon.bounds[3] - polygon.bounds[1]
    xi, yi = np.mgrid[polygon.bounds[0]:polygon.bounds[2]:complex(0, bins),
                      polygon.bounds[1]:polygon.bounds[3]:complex(0, bins)]

    # Perform kernel density estimation only on points inside the polygon
    positions = np.vstack([xi.ravel(), yi.ravel()])
    kernel = gaussian_kde(np.vstack([x, y]), weights=z, bw_method="silverman")

    zi = np.reshape(kernel(positions).T, xi.shape)

    # Create a mask for points inside the polygon
    mask = polygon_path.contains_points(np.column_stack((xi.ravel(), yi.ravel()))).reshape(xi.shape)

    # Apply the mask
    zi_masked = np.ma.masked_where(~mask, zi)


    flattened_valid_values = zi_masked.compressed()
    upper_lim_quant = np.quantile(flattened_valid_values, 0.99)
    # upper_lim_quant = np.max(flattened_valid_values)

    scaled_zi = zi / upper_lim_quant
    zi_masked = np.ma.masked_where(~mask, scaled_zi)
    
    # Plot the contour
    if ax is None:
        fig, ax = plt.subplots()

    if combined_obj is not None:
        plot_str_from_combined_obj(combined_obj, ax=ax)

    cmap_id="viridis"
    # Plot the contour lines
    if color is None:
        # contour = ax.contour(xi, yi, zi_masked, levels=levels, cmap='viridis')
        # contour = ax.contour(xi, yi, zi_masked, levels=levels, cmap='plasma', alpha=1, vmax=upper_lim_quant, vmin=0)
        contour = ax.contour(xi, yi, zi_masked, levels=levels, cmap=cmap_id, alpha=1, vmax=1, vmin=0)
    else:
        contour = ax.contour(xi, yi, zi_masked, levels=levels, colors=color)

    try:
        contour.set_rasterized(True)
    except AttributeError:
        for coll in contour.collections:
            coll.set_rasterized(True)

    if show_color_bar:
        norm = Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap_id, norm=norm)
        sm.set_array([])
        # Add the colorbar
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)
    


    # Plot the polygon boundary
    polygon_points = polygon.exterior.xy
    # ax.plot(*polygon_points, color='black', linewidth=2)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_aspect('equal')

    if return_quant_res:
        res = {"x": xi, "y": yi, "z": zi_masked}
        return res
    
    return ax, contour


def plot_str_from_combined_obj(combined_obj, ax=None, n_points_to_sample = 300000, coords_dict_key = "bead_xy_d", matrix_name="smoothed_matrix", show_legend=False, save_path=None):
    if ax is None:
        fig, ax = plt.subplots()

    # dark_green_code = "#013220"
    gene_color_dict = {"Penk": "DarkGreen", "Mbp": "DarkOrange", "Rarres2": "Purple", "None": "White"}
    alpha_vals_genes = {"Penk": 0.15, "Mbp": 0.15, "Rarres2": 1, "None": 0.01}
    
    # gene_color_dict_query = {"Penk": "DarkGreen", "Mbp": "DarkOrange", "Rarres2": "black", "None": "White"}
    # order in which assigning a color to a bead
    gene_priority = ["Mbp", "Penk", "Rarres2"]
    gene_expression = {}
    for gene in gene_priority:
        gene_expression[gene] = combined_obj[matrix_name][combined_obj["features_idx_dict"][gene], :].toarray().flatten()
    bead_color_lis = ["None"] * combined_obj[matrix_name].shape[1]
    for gene in gene_priority:
        bead_color_lis = [gene if gene_expression[gene][i] > 0 else bead_color_lis[i] for i in range(len(combined_obj["barcodes"]))]
    
    bead_xy_d = combined_obj[coords_dict_key]
    beads = combined_obj["barcodes"]
    
    assert np.all([bead in bead_xy_d for bead in beads])
    bead_xy_d = combined_obj[coords_dict_key]
    beads = combined_obj["barcodes"]
    xyd_stack = list(bead_xy_d.values())
    xyd_name = list(bead_xy_d.keys())
    
    # Extract the x, y, and intensity columns
    query_x = [bead_xy_d[b][0] for b in beads]
    query_y = [bead_xy_d[b][1] for b in beads]
    
    bead_colors = [gene_color_dict[gene] for gene in bead_color_lis]
    bead_alpha_vals = [alpha_vals_genes[gene] for gene in bead_color_lis]
    # query = ax.scatter(query_x, query_y, c=bead_colors,alpha=bead_alpha_vals, s=1, label='Puck')
    assert len(query_x) == len(query_y)
    assert len(query_x) == len(bead_colors)
    assert len(query_x) == len(bead_alpha_vals)
    

    if n_points_to_sample is not None:
        n_beads = len(query_x)
        sample_idx = np.random.choice(n_beads, n_points_to_sample, replace=False)
        query_x = np.array(query_x)[sample_idx]
        query_y = np.array(query_y)[sample_idx]
        bead_colors = np.array(bead_colors)[sample_idx]
        bead_alpha_vals = np.array(bead_alpha_vals)[sample_idx]
    
    
    sc = ax.scatter(query_x, query_y, c=bead_colors, alpha=bead_alpha_vals, s=0.1, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # plt.plot(*str_polygon.exterior.xy,color='blue')
    if show_legend:
        fig1, ax1 = plt.subplots()
        legend_handles = [mpatches.Patch(color=color, label=gene) 
                          for gene, color in gene_color_dict.items() 
                          if gene != "None"]
        ax1.grid(False)
        # Add the legend to the plot
        ax1.legend(handles=legend_handles, loc='upper right', frameon=False)
        # fig1.savefig("legend.pdf")
    if save_path is not None:
        ax.set_rasterization_zorder(100000000000)
        # p = os.path.join(plots_dir, "ventral_med_collat.pdf")
        fig.savefig(save_path, dpi=dpi)


def plot_str_from_combined_obj_sig_beads(combined_obj, ax=None, n_points_to_sample = 300000, coords_dict_key = "bead_xy_d", matrix_name="smoothed_matrix", bead_to_p_val_dict=None, polygon_interest=None, red_sig_normal_other=True, red_sig_gray_other=False, sig_val=0.05):
    if ax is None:
        fig, ax = plt.subplots()
        
    gene_color_dict = {"Penk": "DarkGreen", "Mbp": "DarkOrange", "Rarres2": "Purple", "None": "White"}
    alpha_vals_genes = {"Penk": 0.15, "Mbp": 0.15, "Rarres2": 1, "None": 0.01}
    
    gene_color_dict_query = {"Penk": "DarkGreen", "Mbp": "DarkOrange", "Rarres2": "black", "None": "White"}
    # order in which assigning a color to a bead
    gene_priority = ["Mbp", "Penk", "Rarres2"]
    gene_expression = {}
    for gene in gene_priority:
        gene_expression[gene] = combined_obj[matrix_name][combined_obj["features_idx_dict"][gene], :].toarray().flatten()
    bead_color_lis = ["None"] * combined_obj[matrix_name].shape[1]
    for gene in gene_priority:
        bead_color_lis = [gene if gene_expression[gene][i] > 0 else bead_color_lis[i] for i in range(len(combined_obj["barcodes"]))]
    
    bead_xy_d = combined_obj[coords_dict_key]
    beads = combined_obj["barcodes"]

    assert np.all([bead in bead_xy_d for bead in beads])
    
    bead_p_val = []
    bead_in_poygon = []
    for bead_inx, bead in enumerate(beads):
        bead_point = Point(bead_xy_d[bead][0], bead_xy_d[bead][1])
        bead_in_poygon.append(polygon_interest.contains(bead_point))
        
        bead_p_val.append(bead_to_p_val_dict[bead])
    print(np.mean(bead_in_poygon))


    # Extract the x, y, and intensity columns
    query_x = [bead_xy_d[b][0] for b in beads]
    query_y = [bead_xy_d[b][1] for b in beads]

    if red_sig_normal_other:
        bead_colors = [get_color(c_pval, 0.2) if (c_bead_in_poly and c_pval < sig_val) else gene_color_dict[gene] for gene, c_pval, c_bead_in_poly in zip(bead_color_lis, bead_p_val, bead_in_poygon)]
        bead_alpha_vals = [alpha_vals_genes[gene] for gene in bead_color_lis]
    
    # query = ax.scatter(query_x, query_y, c=bead_colors,alpha=bead_alpha_vals, s=1, label='Puck')
    assert len(query_x) == len(query_y)
    assert len(query_x) == len(bead_colors)
    assert len(query_x) == len(bead_alpha_vals)
    

    if n_points_to_sample is not None:
        n_beads = len(query_x)
        sample_idx = np.random.choice(n_beads, n_points_to_sample, replace=False)
        query_x = np.array(query_x)[sample_idx]
        query_y = np.array(query_y)[sample_idx]
        bead_colors = np.array(bead_colors)[sample_idx]
        bead_alpha_vals = np.array(bead_alpha_vals)[sample_idx]
    
    
    sc = ax.scatter(query_x, query_y, c=bead_colors, alpha=bead_alpha_vals, s=0.1, rasterized=True)
    ax.set_aspect('equal')



def plot_str_spline_from_bead_color_dict(combined_obj, polygon_interest, bead_to_color_dict, coords_dict_key = "bead_xy_d", matrix_name="smoothed_matrix", n_points_to_sample=300000, ax=None, vmin=-0.5, vmax=0.5, cmap=None):
    if ax is None:
        fig, ax = plt.subplots()
    gene_color_dict = {"Penk": "DarkGreen", "Mbp": "DarkOrange", "Rarres2": "Purple", "None": "White"}
    # alpha_vals_genes = {"Penk": 0.1, "Mbp": 0.1, "Rarres2": 1, "None": 0.01}
    alpha_vals_genes = {"Penk": 0.2, "Mbp": 0.2, "Rarres2": 1, "None": 0.01}
    
    gene_color_dict_query = {"Penk": "DarkGreen", "Mbp": "DarkOrange", "Rarres2": "black", "None": "White"}
    # order in which assigning a color to a bead
    gene_priority = ["Mbp", "Penk", "Rarres2"]
    gene_expression = {}
    for gene in gene_priority:
        gene_expression[gene] = combined_obj[matrix_name][combined_obj["features_idx_dict"][gene], :].toarray().flatten()
    bead_color_lis = ["None"] * combined_obj[matrix_name].shape[1]
    for gene in gene_priority:
        bead_color_lis = [gene if gene_expression[gene][i] > 0 else bead_color_lis[i] for i in range(len(combined_obj["barcodes"]))]
    
    bead_xy_d = combined_obj[coords_dict_key]
    beads = combined_obj["barcodes"]
    
    assert np.all([bead in bead_xy_d for bead in beads])
    
    bead_in_poygon = []
    for bead_inx, bead in enumerate(beads):
        bead_point = Point(bead_xy_d[bead][0], bead_xy_d[bead][1])
        bead_in_poygon.append(polygon_interest.contains(bead_point))
    
    # Extract the x, y, and intensity columns
    query_x = [bead_xy_d[b][0] for b in beads]
    query_y = [bead_xy_d[b][1] for b in beads]
    
    bead_colors = [bead_to_color_dict[bead] if (bead_in_polygon_bool and bead in bead_to_color_dict) else gene_color_dict[gene] for gene, bead, bead_in_polygon_bool in zip(bead_color_lis, beads, bead_in_poygon)]
    
    bead_alpha_vals = [alpha_vals_genes[gene] for gene in bead_color_lis]

    # query = ax.scatter(query_x, query_y, c=bead_colors,alpha=bead_alpha_vals, s=1, label='Puck')
    assert len(query_x) == len(query_y)
    assert len(query_x) == len(bead_colors)
    assert len(query_x) == len(bead_alpha_vals)
    
    if n_points_to_sample is not None:
        n_beads = len(query_x)
        sample_idx = np.random.choice(n_beads, n_points_to_sample, replace=False)
        query_x = np.array(query_x)[sample_idx]
        query_y = np.array(query_y)[sample_idx]
        bead_colors = np.array(bead_colors)[sample_idx]
        bead_alpha_vals = np.array(bead_alpha_vals)[sample_idx]
    
    sc = ax.scatter(query_x, query_y, c=bead_colors, alpha=bead_alpha_vals, s=0.1, rasterized=True)
    ax.set_aspect('equal')

def plot_spline_chunked_str(region_result_df, combined_obj, combined_bead_coord_df, polygon_border,cgrid_str_spline_subset, coords_dict_key = "bead_xy_d", region_key = "int_bin", matrix_name="smoothed_matrix", ax=None, point_estimate_col="mean", region_id_col="region", title="STR Enrichment", vmin=-0.35, vmax=0.35, show_ticks=False, start_hex="#ddf4f6", mid_hex = "white",end_hex="#3f2da5", inx_to_line_map={}):

    if ax is None:
        fig, ax = plt.subplots()


    mac_lac_colormap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap',
        [start_hex, mid_hex, end_hex],
        N=256
    )
    # mac_lac_colormap = cm.get_cmap('viridis', n_color_chunks)
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=mac_lac_colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)

    region_result_df["color"] = region_result_df[point_estimate_col].apply(lambda x: value_to_color_hex(x, mac_lac_colormap, norm))
    region_result_df.index = region_result_df[region_id_col]

    region_to_score_dict =  region_result_df[point_estimate_col].to_dict()
    region_to_color_dict = region_result_df["color"].to_dict()

    combined_bead_coord_df["mac_lac_color"] = combined_bead_coord_df[region_key].apply(lambda x: region_to_color_dict[x])

    bead_to_color_dict = combined_bead_coord_df["mac_lac_color"].to_dict()

    plot_str_spline_from_bead_color_dict(combined_obj, polygon_border, bead_to_color_dict, vmin=vmin, vmax=vmax, cmap=mac_lac_colormap, ax=ax)
    
    grouped = cgrid_str_spline_subset.groupby('r_inx_reset')

    
    n_groups_total = len(grouped)
    for iter_inx, (r_inx, group) in enumerate(grouped):
        
        # place p value at top of line
        if iter_inx > -1:
            line = LineString(zip(group['x'], group['y']))

            c_line_obj = inx_to_line_map.get(iter_inx, {})
            c_color = c_line_obj.get("color", "black")
            c_width =  c_line_obj.get("width", 1)
            # Clip the line with the polygon
            clipped_line = line.intersection(polygon_border)
            ax.plot(*clipped_line.xy, color=c_color, linewidth=c_width)
    
            if r_inx not in region_to_score_dict:
                continue
            
            ax.plot(*polygon_border.exterior.xy, color='black', linestyle=':', linewidth=1.75)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_title(title)


def compute_all_zone_enrichments(vt_df, beads_per_chunk, group_col, source_group_sizes, group_scoring_dict, score_col="prob_attr", scale_by_source=True):
    df_results = {}
    for region, vts in beads_per_chunk.items():
        is_in_set = vt_df["vt"].isin(vts)
        matching_vt_df = vt_df[is_in_set].copy()
        scores_df = compute_group_enrichment_2(matching_vt_df, group_col, source_group_sizes, group_scoring_dict, score_col=score_col, scale_by_source=scale_by_source)
        df_results[region] = scores_df
    return df_results


def compute_confidence_interval_df_list(result, n_boot=10000, score_col="weighted_group_score"):
    condensed_results_data_dict = {"region": [], "mean": [], "lower_quantile": [], "higher_quantile": [], "std": [], "lower_error": [], "upper_error": []}
    
    for region, scores_obj in result.items():
        enrich_scores_df = scores_obj['df']
        scores_vec = np.array(enrich_scores_df[score_col])
        n_vals = len(scores_vec)
        enrichment_scores = []
        for iter in range(n_boot):
            sampling = np.random.choice(n_vals, n_vals, replace=True)
            sampled_scores = scores_vec[sampling]
            mean = np.mean(sampled_scores)
            enrichment_scores.append(mean)
        
        mean_enrich = np.mean(enrichment_scores)
        low_quantile = np.quantile(enrichment_scores, q=0.025)
        high_quantile = np.quantile(enrichment_scores, q=0.975)
        std_enrich = np.std(enrichment_scores)

        condensed_results_data_dict["region"].append(region)
        condensed_results_data_dict["mean"].append(mean_enrich)
        condensed_results_data_dict["lower_quantile"].append(low_quantile)
        condensed_results_data_dict["higher_quantile"].append(high_quantile)
        condensed_results_data_dict["std"].append(std_enrich)

        lower_error = mean_enrich - low_quantile
        upper_error = high_quantile - mean_enrich
        condensed_results_data_dict["lower_error"].append(lower_error)
        condensed_results_data_dict["upper_error"].append(upper_error)
    res_df = pd.DataFrame(condensed_results_data_dict)
    return res_df

def compute_group_enrichment_2(matching_vts_df, group_col, group_size_dict, group_score_dict, score_col, scale_by_source=True):
    active_score_col="group_score"
    matching_vts_df["group_score"] = matching_vts_df[group_col].apply(lambda x: group_score_dict[x])
    
    if scale_by_source:
        total_weight = np.sum(list(group_size_dict.values()))
        matching_vts_df["group_score_scaled"] = matching_vts_df.apply(lambda row: row[active_score_col] / group_size_dict[row[group_col]], axis=1) 
        matching_vts_df["group_score_scaled"] = matching_vts_df["group_score_scaled"] * total_weight
        active_score_col="group_score_scaled"
    
    matching_vts_df["weighted_group_score"] = matching_vts_df[score_col] * matching_vts_df[active_score_col]
    mean_weighted_score = matching_vts_df["weighted_group_score"].mean()
    res = {"df": matching_vts_df, "mean": mean_weighted_score}
    return res


def get_color(p_value, min_p = 0.05):
    if p_value > min_p:
        return "#808080"  # Gray for p-values > 0.1
    else:
        # Create a red spectrum for p-values <= 0.1
        # Map p-value from [0, 0.1] to [255, 100] for red intensity
        red = 255
        max_scale = 0.15
        # Adjust the range of green and blue to create a more vibrant gradient
        green_blue = int(100 + (155 * (p_value / max_scale)))
        # Ensure values are within the valid range
        green_blue = max(100, min(255, green_blue))
        return f"#{red:02X}{green_blue:02X}{green_blue:02X}"


def value_to_color_hex(value, cmap, norm=None):

    if norm is not None:
        value = norm(value)
    # Get the RGBA color for this value
    rgba = cmap(value)
    
    # Convert RGBA to hex
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(rgba[0] * 255),
        int(rgba[1] * 255),
        int(rgba[2] * 255)
    )
    
    return hex_color

def df_to_polygon(polygon_df):
    points = []
    for inx, row in polygon_df.iterrows():
        coords = [row["x"], row["y"]]
        points.append(coords)
    polygon = Polygon(points)
    return polygon


subclass_to_id = {
    "007_L2-3_IT_CTX_Glut": "L2/3 IT",
    "006_L4-5_IT_CTX_Glut": "L4/5 IT",
    "005_L5_IT_CTX_Glut": "L5 IT",
    "004_L6_IT_CTX_Glut": "L6 IT",
    "022_L5_ET_CTX_Glut": "L5 ET",
    "030_L6_CT_CTX_Glut": "L6 CT",
    "032_L5_NP_CTX_Glut": "L5 NP",
    "029_L6b_CTX_Glut": "L6b",
    "Inh": "Inh"
}

def compute_difference_mean_enrich_results(df1, df2, score_col="weighted_group_score"):
    n_boot = 10000
    group1_mean = df1[score_col].mean()
    group2_mean = df2[score_col].mean()
    combined_dist = list(df1[score_col]) + list(df2[score_col])
    combined_mean = np.mean(combined_dist)
    observed_diff = group1_mean - group2_mean
    
    t_observed = scipy.stats.ttest_ind(df1[score_col], df2[score_col], equal_var=False)
    t_stat = t_observed.statistic
    df = t_observed.df
    group_1_dist_mid_set = df1[score_col] - group1_mean + combined_mean
    group_2_dist_mid_set = df2[score_col] - group2_mean + combined_mean
    n_group1 = len(group_1_dist_mid_set)
    n_group2 = len(group_2_dist_mid_set)
    bootstrapped_t_vals = []
    for iter in range(n_boot):
        samples_group_1 = np.random.choice(group_1_dist_mid_set, n_group1, replace=True)
        samples_group_2 = np.random.choice(group_2_dist_mid_set, n_group2, replace=True)
    
        iter_t = scipy.stats.ttest_ind(samples_group_1, samples_group_2, equal_var=False)
        current_t = iter_t.statistic
        bootstrapped_t_vals.append(current_t)
    p_val = np.mean(np.array(bootstrapped_t_vals) > t_stat)

    res = {"group_1_mean": group1_mean, "group_2_mean": group2_mean, "alt": "group 1 > group 2", "p": p_val, "t_stat": t_stat, "df": df}
    return res



def preform_all_comparisons(region_enrich_results, score_col="weighted_group_score", n_adj=1, show_both_sides=False):

    print(score_col)
    results_data_dict = {"group_1": [], "group_2": [], "group_1_mean": [], "group_2_mean": [], "alt_1": [], "p_1": [], "t": [], "df": []}
    if show_both_sides:
        results_data_dict["alt_2"] = []
        results_data_dict["p_2"] = []
    
    all_groups = region_enrich_results.keys()
    for group_1_inx, group_1_name in enumerate(all_groups):
        print(group_1_name)
        for group_2_inx, group_2_name in enumerate(all_groups):
            if group_1_inx >= group_2_inx:
                continue
            group_1_df = region_enrich_results[group_1_name]["df"]
            group_2_df = region_enrich_results[group_2_name]["df"]

            compar_res = compute_difference_mean_enrich_results(group_1_df, group_2_df, score_col=score_col)
            results_data_dict["group_1"].append(group_1_name)
            results_data_dict["group_2"].append(group_2_name)
            results_data_dict["group_1_mean"].append(compar_res["group_1_mean"])
            results_data_dict["group_2_mean"].append(compar_res["group_2_mean"])
            
            results_data_dict["alt_1"].append(compar_res["alt"])
            results_data_dict["p_1"].append(compar_res["p"])
            results_data_dict["t"].append(compar_res["t_stat"])
            results_data_dict["df"].append(compar_res["df"])

    
    res_df = pd.DataFrame(results_data_dict)
    res_df["p_1_adj"] = res_df["p_1"] * n_adj
    return res_df



VT_OBJ_ID_CONFIGS = {
    "v1": {
        "region_ordering": ["PG", "TH", "SC", "CP", "VISPc"],
        "cell_type_ordering": ["L4/5 IT", "L5 IT", "L6 IT", "L5 ET", "L6 CT", "L5 NP", "Inh"],
        "subclasses_keep": ["006_L4-5_IT_CTX_Glut", "005_L5_IT_CTX_Glut", "004_L6_IT_CTX_Glut", "022_L5_ET_CTX_Glut", "030_L6_CT_CTX_Glut", "032_L5_NP_CTX_Glut", "029_L6b_CTX_Glut", "Inh", "null"],
    },
    "aca": {
        "region_ordering": ["cMed", "STN", "TH", "STR", "cMOp"],
        "cell_type_ordering": ["L2/3 IT", "L4/5 IT", "L5 IT", "L6 IT", "L5 ET", "L6 CT", "L6b", "L5 NP", "Inh"],
        "subclasses_keep": ["007_L2-3_IT_CTX_Glut", "006_L4-5_IT_CTX_Glut", "005_L5_IT_CTX_Glut", "004_L6_IT_CTX_Glut", "022_L5_ET_CTX_Glut", "030_L6_CT_CTX_Glut", "029_L6b_CTX_Glut", "032_L5_NP_CTX_Glut", "Inh", "null"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run projection analysis on a VT scoring dataframe.")
    parser.add_argument("--vt_df_path", type=str,
                        default="../objects/vt_processing/v1_vt_df.csv",
                        help="Path to the input vt_df CSV.")
    parser.add_argument("--obj_dir", type=str,
                        default="../objects/vt_processing",
                        help="Directory containing input processing objects.")
    parser.add_argument("--vt_obj_id", type=str, default="v1",
                        choices=list(VT_OBJ_ID_CONFIGS.keys()),
                        help="VT object identifier (controls region/cell-type orderings).")
    parser.add_argument("--out_dir_base", type=str,
                        default="../figs",
                        help="Base output directory. A subdirectory named after the input file is created inside.")
    parser.add_argument("--aca_vt_score_path", type=str,
                        default=None,
                        help="Path to aca scoring df, if running aca branch")

    return parser.parse_args()


class _PdfAndSvgSaver:


    def __init__(self, pdf, svg_dir):
        self._pdf = pdf
        self._svg_dir = svg_dir
        self._counter = 0
        os.makedirs(svg_dir, exist_ok=True)

    def savefig(self, fig, name=None, **kwargs):
        self._pdf.savefig(fig, **kwargs)
        svg_kwargs = {k: kwargs[k] for k in ("transparent", "bbox_inches", "dpi") if k in kwargs}
        svg_name = name if name is not None else f"fig_{self._counter:02d}"
        svg_path = os.path.join(self._svg_dir, f"{svg_name}.svg")
        fig.savefig(svg_path, format="svg", **svg_kwargs)
        self._counter += 1


def run(vt_df_path, obj_dir, vt_obj_id, out_dir_base, aca_vt_score_path=None):

    print(f"VT Score Path: {aca_vt_score_path}")

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    min_ct_n = 100
    NBOOT = 1000

    cfg = VT_OBJ_ID_CONFIGS[vt_obj_id]
    region_ordering = cfg["region_ordering"]
    cell_type_ordering = cfg["cell_type_ordering"]
    subclasses_keep = cfg["subclasses_keep"]

    out_prefix = os.path.basename(vt_df_path).split(".csv")[0]
    out_dir = os.path.join(out_dir_base, out_prefix)
    os.makedirs(out_dir, exist_ok=True)

    leaf_to_subclass_dict_path = os.path.join(obj_dir, LEAF_TO_SUBCLASS_FILENAME)

    leaf_to_subclass_dict = pickle.load(open(leaf_to_subclass_dict_path, "rb"))

    leaf_to_subclass_dict_inh_grouped = {}
    for leaf, subclass in leaf_to_subclass_dict.items():
        if leaf.startswith("Inh"):
            leaf_to_subclass_dict_inh_grouped[leaf] = "Inh"
        else:
            leaf_to_subclass_dict_inh_grouped[leaf] = subclass

    unique_layer_groups = np.unique(list(leaf_to_subclass_dict_inh_grouped.values()))
    layer_to_leaf_dict = {layer: [leaf for leaf in leaf_to_subclass_dict_inh_grouped if leaf_to_subclass_dict_inh_grouped[leaf] == layer] for layer in unique_layer_groups}
    layer_to_leaf_tup_list = [(layer, layer_to_leaf_dict[layer]) for layer in unique_layer_groups]
    layer_to_leaf_tup_list.append(("null", ["Glia", "empty", "unassigned", "Inh"]))

    projection_objs_path = os.path.join(obj_dir, PROJECTION_TARGETS_FILENAME)
    projection_objs = pickle.load(open(projection_objs_path, "rb"))

    proj_targets = projection_objs.get(vt_obj_id, None)
    if proj_targets is None:
        raise ValueError(f"vt_obj_id '{vt_obj_id}' not found in {projection_objs_path}")

    cell_vt_df = pd.read_csv(vt_df_path)
    cell_vt_leaf_passing = cell_vt_df.loc[cell_vt_df["pknn_pass_rate"] == 1]

    subclass_visual_cell_match_rates_dict = create_type_to_region_prop_cell_match(cell_vt_df, proj_regions=proj_targets, cell_type_col="subclass_visual")
    leaf_cell_match_rates_dict = create_type_to_region_prop_cell_match(cell_vt_leaf_passing, proj_regions=proj_targets, cell_type_col=cluster_grouped_col)

    subclass_projs = preform_all_celltype_region_comparisons_jackknife(cell_vt_df, proj_targets, score_col="prob_attr", cell_type_col="subclass_visual", min_ct_n=min_ct_n, null_type="Glia", jackknife_holdout_prop=0.1, NBOOT=NBOOT)
    leaf_projs = preform_all_celltype_region_comparisons_jackknife(cell_vt_leaf_passing, proj_targets, score_col="prob_attr", cell_type_col=cluster_grouped_col, min_ct_n=min_ct_n, null_type="Glia", jackknife_holdout_prop=0.1, NBOOT=NBOOT)

    subclass_to_leaf_dict = {ele[0]: ele[1] for ele in layer_to_leaf_tup_list}
    subclass_to_leaf_tup_ordered = [(subclass, subclass_to_leaf_dict[subclass]) for subclass in subclasses_keep]

    leaf_projs["subclass"] = leaf_projs["cell_type"].apply(lambda x: leaf_to_subclass_dict.get(x, x))
    leaf_projs["subclass_visual"] = leaf_projs["subclass"].apply(lambda x: subclass_to_id.get(x, x))
    leaf_projs["class"] = leaf_projs["cell_type"].apply(lambda x: x.split("_")[0])
    leaf_projs.loc[leaf_projs["class"] == "Inh", "subclass_visual"] = "Inh"
    leaf_projs_subclass_interest = subset_to_valid_subclasses(leaf_projs, cell_type_ordering)

    combined_pdf_out = os.path.join(out_dir, f"{out_prefix}_projections.pdf")
    svg_out_dir = os.path.join(out_dir, "svg")

    with PdfPages(combined_pdf_out) as _pdf:
        pdf = _PdfAndSvgSaver(_pdf, svg_out_dir)
        fig = create_basic_dotplot_proj_sub_groupings_continous(subclass_projs,
                                    cell_type_col="cell_type",
                                    region_col="region",
                                    color_col="prop_vts",
                                    text_col="jack_p_val",
                                    value_col="jack_p_val",
                                    sig_val=0.05,
                                    negate_sig_val=True, only_show_text_if_sig=False,
                                    only_show_projecting_cts=False,
                                    title="Subclass Region Projections",
                                    size_scale=1, show_text=False,
                                    space_per_ct=0.6, space_per_region=1.1,
                                    region_ordering=region_ordering,
                                    cell_type_ordering=cell_type_ordering,
                                    sizing_factor_y=0.1,
                                    sizing_factor_x=0.05,
                                    sig_circle_width=3,
                                    cell_match_rates_dict=subclass_visual_cell_match_rates_dict,
                                    vmax=0.25)
        pdf.savefig(fig, transparent=True, bbox_inches="tight")

        fig = create_basic_dotplot_proj_sub_groupings_continous(leaf_projs_subclass_interest,
                                    cell_type_col="cell_type",
                                    region_col="region",
                                    color_col="prop_vts",
                                    text_col="jack_p_val",
                                    value_col="jack_p_val",
                                    sig_val=0.05,
                                    negate_sig_val=True,
                                    only_show_text_if_sig=True,
                                    only_show_projecting_cts=False,
                                    title="Leaf Types Region Projections",
                                    size_scale=1, show_text=False,
                                    x_axis_grouping_tup_list=subclass_to_leaf_tup_ordered,
                                    space_per_ct=0.55, space_per_region=2.5,
                                    show_intersections_by_ct=True,
                                    region_ordering=region_ordering,
                                    sizing_factor_x=-0.035,
                                    min_n_red_circle=0,
                                    xtick_fsize=16,
                                    cell_match_rates_dict=leaf_cell_match_rates_dict,
                                    vmax=0.25,
                                    sig_circle_width=2)
        pdf.savefig(fig, transparent=True, bbox_inches="tight")

        if vt_obj_id == "v1":
            cp_only_projs = projection_objs["cp_sub_dissects"]
            subclass_cp_projs_match_rates = create_type_to_region_prop_cell_match(cell_vt_df, proj_regions=cp_only_projs, cell_type_col="subclass_visual")
            subclass_cp_projs = preform_all_celltype_region_comparisons_jackknife(cell_vt_df, cp_only_projs, score_col="prob_attr", cell_type_col="subclass_visual", min_ct_n=min_ct_n, null_type="Glia", jackknife_holdout_prop=0.1, NBOOT=NBOOT)
            fig = create_basic_dotplot_proj_sub_groupings_continous(subclass_cp_projs,
                                        cell_type_col="cell_type",
                                        region_col="region",
                                        color_col="prop_vts",
                                        text_col="jack_p_val",
                                        value_col="jack_p_val",
                                        sig_val=0.05,
                                        negate_sig_val=True, only_show_text_if_sig=False,
                                        only_show_projecting_cts=False,
                                        show_intersections_by_ct=False,
                                        title="Subclass Region Projections CP",
                                        size_scale=1, show_text=False,
                                        space_per_ct=0.5, space_per_region=0.5,
                                        sizing_factor_y=0.12,
                                        sizing_factor_x=0.03,
                                        cell_type_ordering=cell_type_ordering,
                                        cell_match_rates_dict=subclass_cp_projs_match_rates,
                                        vmax=0.03, sig_circle_width=2)
            pdf.savefig(fig, transparent=True, bbox_inches="tight")

            # thalamus specific projections
            # thalamus_proj_path = os.path.join(obj_dir, THALAMUS_SUBSECTION_PROJECTIONS_FILENAME)
            # thalamus_proj_regions = pickle.load(open(thalamus_proj_path, "rb"))
            thalamus_proj_regions = projection_objs.get("TH_subnuclei", None)
            if thalamus_proj_regions is None:
                raise Exception("TH sub nuclei not found in projection target object")

            subclass_interest = ["L5 ET", "L6 CT", "Glia", "Inh"]
            min_passing_thresh = 1
            is_passing = cell_vt_df["pknn_pass_rate"] >= min_passing_thresh
            is_subclass = cell_vt_df["subclass_visual"].isin(subclass_interest)
            is_passing_et_ct = is_passing & is_subclass
            cell_vt_df_passing_et_ct = cell_vt_df[is_passing_et_ct]

            ct_mapping = {
                "Ex_Foxp2_Col5a1": "L6_CT_1",
                "Ex_Fezf2_Cpa6": "L6_CT_2",
                "Ex_Fezf2_Sebox": "L6_CT_2",
                "Ex_Fezf2_Klhl1": "L5_ET_1",
            }

            is_glia = cell_vt_df_passing_et_ct[cluster_grouped_col] == "Glia"
            is_ct_interest = cell_vt_df_passing_et_ct[cluster_grouped_col].isin(ct_mapping.keys())

            cell_vt_df_passing_et_ct_group_interest = cell_vt_df_passing_et_ct[is_glia | is_ct_interest].copy()
            cell_vt_df_passing_et_ct_group_interest["ct_et_grouped"] = cell_vt_df_passing_et_ct_group_interest[cluster_grouped_col].apply(lambda x: ct_mapping.get(x, x))

            et_ct_thalamic_res = preform_all_celltype_region_comparisons_jackknife(cell_vt_df_passing_et_ct_group_interest, thalamus_proj_regions, score_col="prob_attr", cell_type_col="ct_et_grouped", min_ct_n=0, null_type="Glia", jackknife_holdout_prop=0.1, subset_mscar=True, NBOOT=1000, seed=10, n_jackknives=10000)

            ct_et_subtypes_prop_cells_matching = create_type_to_region_prop_cell_match(cell_vt_df_passing_et_ct_group_interest, proj_regions=thalamus_proj_regions, cell_type_col="ct_et_grouped")

            subclass_to_leaf_tup_ordered_ct = [
                ("L5 ET", ["L5_ET_1", "L5_ET_2", "L5_ET_3"]),
                ("L6 CT", ["L6_CT_1", "L6_CT_2"])
            ]

            fig = create_basic_dotplot_proj_sub_groupings_continous(et_ct_thalamic_res,
                                        cell_type_col="cell_type",
                                        region_col="region",
                                        color_col="prop_vts",
                                        text_col="jack_p_val",
                                        value_col="jack_p_val",
                                        sig_val=0.05,
                                        region_ordering=None,
                                        negate_sig_val=True, only_show_text_if_sig=True,
                                        title="",
                                        min_n_red_circle=1,
                                        min_show_val=0.25,
                                        only_show_projecting_cts=False,
                                        size_scale=1.4, show_text=False, xtick_fsize=16, ytick_fsize=16,
                                        vmax=0.05, show_intersections_by_ct=False,
                                        sizing_factor_y=1.2,
                                        sizing_factor_x=0.15,
                                        x_axis_grouping_tup_list=subclass_to_leaf_tup_ordered_ct,
                                        space_per_ct=1.35, space_per_region=1,
                                        cell_match_rates_dict=ct_et_subtypes_prop_cells_matching)
            pdf.savefig(fig, transparent=True, bbox_inches="tight")

        if vt_obj_id == "aca":
        
            "Load AC specific Objects"
            str_chunked_objs = projection_objs.get("STR_subregion", None)
            if str_chunked_objs is None:
                raise Exception("STR subregions not found in projection object")
            str_chunked_4 = str_chunked_objs[4]

            str_combined_obj_path = os.path.join(obj_dir, STR_COMBINED_OBJ_FILENAME)
            combined_obj = pickle.load(open(str_combined_obj_path, "rb"))

            str_polygon_df = combined_obj["str_coords"]
            str_polygon = df_to_polygon(str_polygon_df)

            cgrid_str_spline_subset_4 = str_chunked_4["cgrid_str_spline_subset"]
            beads_per_spline_chunk_4 = str_chunked_4["vts_per_spline_chunk"]

            str_chunked_2 = str_chunked_objs[2]
            beads_per_spline_chunk_2 = str_chunked_2["vts_per_spline_chunk"]

            combined_bead_coord_df_w_spline_4 = str_chunked_4["combined_bead_coord_df_w_spline"]
            if aca_vt_score_path is not None:
                # aca_vt_scores = pd.read_csv("/home/jsilverm/Working_SynapseSeq/objs/str_bead_locations.csv")
                aca_vt_scores = pd.read_csv(aca_vt_score_path)
                print(aca_vt_scores.head())
            else:
                raise Exception("Must pass in aca scoring path for aca analysis")

            str_vt_merged_p = os.path.join(obj_dir, STR_VT_DF_BEAD_MERGED_FILENAME)
            vt_merged_bead_str_df = pd.read_csv(str_vt_merged_p, index_col=0)

            cell_vt_df_leaf_passing = cell_vt_df.loc[cell_vt_df["pknn_pass_rate"] == 1].copy()
            add_glia_inh_column(cell_vt_df_leaf_passing, cluster_grouped_col)

            str_subclasses_interest = ["L5 IT", "L6 IT", "L4/5 IT", "L5 ET", "Glia", "Inh"]
            cell_vt_df_leaf_passing_subclass_interest = cell_vt_df_leaf_passing.loc[cell_vt_df_leaf_passing["subclass_visual"].isin(str_subclasses_interest)]

            leaf_types_spline_result_4_glia_inh = preform_all_celltype_region_comparisons_jackknife(cell_vt_df_leaf_passing_subclass_interest, beads_per_spline_chunk_4, score_col = "prob_attr", cell_type_col = f"{cluster_grouped_col}_inh_glia", min_ct_n = 200, null_type="Inh-Glia", jackknife_holdout_prop=0.1, NBOOT=5000, subset_mscar=True, n_jackknives=100000)

            min_n = 10
            matches_per_type = leaf_types_spline_result_4_glia_inh.groupby("cell_type")["n_vts_proj_region"].sum().reset_index()
            valid_types = set(matches_per_type.loc[matches_per_type["n_vts_proj_region"] >= min_n, "cell_type"].values)
            leaf_types_valid_str_res = leaf_types_spline_result_4_glia_inh.loc[leaf_types_spline_result_4_glia_inh["cell_type"].isin(valid_types)]
            fig = create_basic_dotplot_proj_sub_groupings_continous(leaf_types_valid_str_res,
                                                cell_type_col="cell_type",
                                                region_col="region",
                                                color_col = "prop_vts",
                                                text_col="jack_p_val",
                                                value_col="jack_p_val",
                                                only_show_projecting_cts=False,
                                                sig_val = 0.05,
                                                negate_sig_val=True,
                                                min_show_val=0.25,
                                                only_show_text_if_sig=True,
                                                space_per_ct = 0.75,
                                                sizing_factor_x=0.1,
                                                sizing_factor_y=0.05,
                                                space_per_region=1,
                                                x_axis_grouping_tup_list = layer_to_leaf_tup_list,
                                                title=f"ET"
                                            )
            pdf.savefig(fig, transparent=True, bbox_inches="tight")

            for ct_name in ["Ex_Rorb_Rxfp2", "Ex_Slc30a3_Sulf1_Aprk1", "Ex_Slc30a3_Col6a1"]:
                fig, ax = plt.subplots()
                plot_subpatch_results(leaf_types_spline_result_4_glia_inh, ct_name, combined_obj, cgrid_str_spline_subset_4, str_polygon, combined_bead_coord_df_w_spline_4, ax=ax)
                ax.set_title(ct_name)
                pdf.savefig(fig, transparent=True, bbox_inches="tight", dpi=dpi)


            region_coord_y = {
            "cMed1": 2,
            "cMed2": 2,
            "cMed8": 1.75,
            
            "cMed3": 1,
            "cMed4": 1,
            
            "cMed7": 0.25,
            "cMed5": 0,
            "cMed6": 0,

            }

            placed_coord_to_group_dict = {
                0: "ventral-med",
                0.25: "ventral-med",
                1: 'mid-med',
                1.75: "dorsal-med",
                2: "dorsal-med"
            }
            
            
            
            med_dissec_to_group = {
                "cMed1": "dorsal-med",
                "cMed2": "dorsal-med",
                "cMed8": "dorsal-med",
            
                "cMed3": "mid-med",
                "cMed4": "mid-med",
            
                "cMed5": "ventral-med",
                "cMed6": "ventral-med",
                "cMed7": "ventral-med"
            }
            

            med_proj_support_p = os.path.join(obj_dir, MEDULLA_DISSECTION_FILENAME)
            med_vt_umi_count_df = pd.read_csv(med_proj_support_p, index_col=0)
            med_vt_umi_count_df["y"] = med_vt_umi_count_df["region"].apply(lambda region: region_coord_y[region])


            add_dissectate_grouping_vars_cell_vt_df(cell_vt_df_leaf_passing)
            med_proj_matching_vt_df = pd.merge(cell_vt_df_leaf_passing, med_vt_umi_count_df, on="vt", how="inner")
            
            et_med_proj_matching_df = med_proj_matching_vt_df.loc[med_proj_matching_vt_df["subclass_visual"] == "L5 ET"]
            
            et_med_proj_matching_df_region_max_umi, region_umi_df_valid = compute_max_umi_region_df(med_vt_umi_count_df, et_med_proj_matching_df, region_col_med="y")
            
            fig, ax = plt.subplots(figsize=(3, 4))
            med_95_conf_interval = plot_group_conf_interval(vt_matching_res=et_med_proj_matching_df_region_max_umi, dissectate_grouping="dissectate_grouped_4", coord_interest="y_placed", score_col="prob_attr", ax=ax, N_BOOT=10000)
            ax.set_ylim(0.25, 1.5)
            ax.set_title("L5 ET medulla group 95% CI")
            pdf.savefig(fig, transparent=True, bbox_inches="tight")


            region_umi_df_valid["dissec_region"] = region_umi_df_valid["y_placed"].apply(lambda x: placed_coord_to_group_dict[x])
            
            aca_vt_scores["vt_score_attr"] = list(map(lambda x: 1 / (1 + x), aca_vt_scores["expected_cell_deliveries"]))
            aca_vt_scores.index = aca_vt_scores["vt"]
            aca_vt_to_score_dict = aca_vt_scores["vt_score_attr"].to_dict()
            
            medulla_str_merged_vts_med_region = pd.merge(region_umi_df_valid, vt_merged_bead_str_df, left_index=True, right_on="vt", how="inner", suffixes=("_medulla", None))
            medulla_str_merged_vts_med_region["vt_score_attr"] = medulla_str_merged_vts_med_region["vt"].apply(lambda x: aca_vt_to_score_dict[x])
            
            medulla_str_merged_vts_med_region["matches"] = True
            min_thresh = 0
            
            medulla_str_merged_vts_bead_filt_thresh = medulla_str_merged_vts_med_region.loc[medulla_str_merged_vts_med_region["vt_score_attr"] > min_thresh]
            
            grouping_col="dissec_region"
            fig, ax = plt.subplots(figsize=(18,6), ncols=3)
            for panel_inx, panel_name in enumerate(["ventral-med", "mid-med", "dorsal-med"]):
                weighted_density_contour(medulla_str_merged_vts_bead_filt_thresh, cell_type_col=grouping_col, cell_type_name=panel_name, weight_col="vt_score_attr", polygon=str_polygon, bins=100, levels=5, combined_obj=combined_obj, ax=ax[panel_inx], show_color_bar=True)
                ax[panel_inx].set_title(panel_name)
            pdf.savefig(fig, name="density_contour", transparent=True, bbox_inches="tight", dpi=dpi)


            med_scoring = {"dorsal-med": 1, "mid-med": 0, "ventral-med": -1}
            med_group_sizes = medulla_str_merged_vts_med_region.groupby("dissec_region").size().to_dict()
            med_group_sum_scores = medulla_str_merged_vts_med_region.groupby("dissec_region")["vt_score_attr"].sum().to_dict()
            
            region_enrich_results_4 = compute_all_zone_enrichments(medulla_str_merged_vts_med_region, beads_per_spline_chunk_4, "dissec_region", med_group_sum_scores, med_scoring, "vt_score_attr", scale_by_source=True)
            boot_results_4 = compute_confidence_interval_df_list(region_enrich_results_4, n_boot=1000)
            inx_to_line_map = {
                    0: {"color": "black", "width": 0.5},
                    1: {"color": "black", "width": 2},
                    2: {"color": "black", "width": 0.5}
                }
            
            combined_bead_coord_df_w_spline_4.index = combined_bead_coord_df_w_spline_4["region_bead"]
            
            med_ventral_hex="#FF0000"
            med_mid_hex = "white"
            med_dorsal_hex="#0000FF"
            fig, ax = plt.subplots(figsize=(6,6))
            plot_spline_chunked_str(boot_results_4, combined_obj, combined_bead_coord_df_w_spline_4, polygon_border=str_polygon, cgrid_str_spline_subset=cgrid_str_spline_subset_4, title="", start_hex=med_ventral_hex, end_hex=med_dorsal_hex, vmax=0.45, vmin=-0.45, ax=ax, inx_to_line_map=inx_to_line_map)
            pdf.savefig(fig, name="spline_chunked_str", transparent=True, bbox_inches="tight", dpi=dpi)


            region_enrich_results_2 = compute_all_zone_enrichments(medulla_str_merged_vts_med_region, beads_per_spline_chunk_2, "dissec_region", med_group_sum_scores, med_scoring, "vt_score_attr", scale_by_source=True)
            boot_results_2 = compute_confidence_interval_df_list(region_enrich_results_2, n_boot=10000)
            
            diff_res_4_2 = preform_all_comparisons(region_enrich_results_2)
            out_p = os.path.join(out_dir, "collat_p_val.pkl")
            pickle.dump(diff_res_4_2, open(out_p, "wb"))


if __name__ == "__main__":
    args = parse_args()
    run(
        vt_df_path=args.vt_df_path,
        obj_dir=args.obj_dir,
        vt_obj_id=args.vt_obj_id,
        out_dir_base=args.out_dir_base,
        aca_vt_score_path = args.aca_vt_score_path
    )

