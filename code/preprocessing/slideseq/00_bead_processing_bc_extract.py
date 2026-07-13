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

import slideseq.bead_matching
from slideseq.plot import new_ax
from slideseq.util.logger import create_logger

log = logging.getLogger("barcode_matrix")

#DSWHSWVSWBSWDSWSWVSWBSWDSWHSWVSW

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


def plot_log_hist(dist, title: str, pdf_pages: PdfPages):
    max_d = np.ceil(np.log10(max(dist)))
    with new_ax(pdf_pages) as ax:
        ax.hist(dist, bins=np.logspace(0, max_d, int(max_d * 10 + 1)), log=True)
        ax.set_xscale("log")
        ax.set_title(title)


def plot_hist(dist, title: str, pdf_pages: PdfPages):
    with new_ax(pdf_pages) as ax:
        ax.hist(dist, bins=100, log=True)
        ax.set_title(title)


def spatial_plot(bead_xy_a, dist, title: str, pdf_pages: PdfPages, pct: float = 95.0):
    with new_ax(pdf_pages, include_fig=True) as (fig, ax):
        # version of 'Blues' colormap that is pure white at the bottom
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "BluesW", [(1.0, 1.0, 1.0), (0.0314, 0.188, 0.450)]
        )
        norm = matplotlib.colors.Normalize(0, np.percentile(dist, pct), clip=True)

        c = ax.scatter(
            bead_xy_a[:, 0],
            bead_xy_a[:, 1],
            c=dist,
            s=0.5,
            cmap=cmap,
            norm=norm,
        )
        c.set_rasterized(True)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.set_title(title)
        fig.colorbar(c, ax=ax)


def spatial_plot_plus_links(
    bead_xy_a, bead_pairs, umi_dist, title: str, pdf_pages: PdfPages, pct: float = 95
):
    with new_ax(pdf_pages, include_fig=True) as (fig, ax):
        # version of 'Blues' colormap that is pure white at the bottom
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "BluesW", [(1.0, 1.0, 1.0), (0.0314, 0.188, 0.450)]
        )
        norm = matplotlib.colors.Normalize(0, np.percentile(umi_dist, pct), clip=True)

        c = ax.scatter(
            bead_xy_a[:, 0],
            bead_xy_a[:, 1],
            c=umi_dist,
            s=0.5,
            cmap=cmap,
            norm=norm,
        )
        c.set_rasterized(True)

        xs = bead_xy_a[bead_pairs, 0].T
        ys = bead_xy_a[bead_pairs, 1].T

        ax.plot(xs, ys, alpha=0.1, color="g", linewidth=0.1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        ax.set_title(title)
        fig.colorbar(c, ax=ax)

def get_barcodes(puck_dir: Path):
    log.info(f"Reading bead data from {puck_dir}")
    log.debug(f"Reading {puck_dir / 'BeadBarcodes.txt'}")
    with open(puck_dir / "BeadBarcodes.txt") as fh:
        raw_bcs = ["".join(line.strip().split(",")) for line in fh]

    log.debug(f"Reading {puck_dir / 'BeadLocations.txt'}")
    with open(puck_dir / "BeadLocations.txt") as fh:
        x = np.array([float(v) for v in fh.readline().strip().split(",")])
        y = np.array([float(v) for v in fh.readline().strip().split(",")])
        xy = np.vstack((x, y)).T

    ok_barcodes = [not set(bc).issubset({"T", "N"}) for bc in raw_bcs]
    xy = xy[ok_barcodes, :]
    bead_barcodes = [bc for ok, bc in zip(ok_barcodes, raw_bcs) if ok]

    log.info(f"Read {len(raw_bcs)} barcodes and filtered to {len(bead_barcodes)}")

    return bead_barcodes, xy

def bead_network(bead_barcodes, seq_barcodes, xy):
    # adjacency matrix for all beads within radius of each other
    radius_matrix = radius_neighbors_graph(xy, radius=10.0)

    # adjacency matrix for all barcodes within hamming distance 1
    hamming_matrix = slideseq.bead_matching.hamming1_adjacency(bead_barcodes)

    # just multiply together to get the combined adjacency matrix!
    combined_graph = nx.from_scipy_sparse_matrix(radius_matrix.multiply(hamming_matrix))

    # add xy coordinates to graph so we can analyze later
    for n, (x, y) in zip(combined_graph.nodes, xy):
        combined_graph.nodes[n]["x"] = x
        combined_graph.nodes[n]["y"] = y

    # get connected components to find groups of similar/close barcodes
    bead_groups = list(nx.connected_components(combined_graph))

    # calculate degenerate (ambiguous bases -> N) barcodes
    degen_bead_barcodes = [
        slideseq.bead_matching.degen_barcode({bead_barcodes[j] for j in bg})
        for bg in bead_groups
    ]

    log.debug(
        f"Collapsed {len(bead_groups)} bead groups into"
        f" {len(set(degen_bead_barcodes))} barcodes"
    )

    # average xy for grouped beads to get centroids
    bead_xy = dict()
    for bg, degen_bc in zip(bead_groups, degen_bead_barcodes):
        bg_graph = combined_graph.subgraph(bg)
        mean_x, mean_y = np.array(
            [[nd["x"], nd["y"]] for _, nd in bg_graph.nodes(data=True)]
        ).mean(0)
        bead_xy[degen_bc] = (mean_x, mean_y)

    barcode_matching = slideseq.bead_matching.bipartite_matching(
        bead_barcodes, degen_bead_barcodes, bead_groups, seq_barcodes
    )

    return degen_bead_barcodes, bead_xy, barcode_matching

def match_tags(
    r1_reads, r2_reads, barcode_mapping, cellID_codes, cellID_cutoff, constant_seq_hd1, constant_seq
):
    false_beads = 0
    no_const = 0
    no_id_struct = 0

    upper_val = 57+len(constant_seq)

    raw_umis_per_bead = defaultdict(lambda: defaultdict(set))
    raw_umis_per_bead_umi = defaultdict(Counter)

    for r1, r2 in zip(r1_reads, r2_reads):
        
        seq_bc = r1[:8] + r1[26:32]
        tag = r2[25:57]


        # Check if barcode is in list of approved barcodes
        if seq_bc not in barcode_mapping:
            false_beads += 1
            continue

        # Check if Cell ID does not contain constant sequence
        if r2[57:upper_val] not in constant_seq_hd1:
            no_const += 1
            continue

        # Check if Cell ID does not match structure
        if sum(b in s for b, s in zip(tag, cellID_codes)) < cellID_cutoff:
            no_id_struct += 1
            continue

        bead_bc = barcode_mapping[seq_bc]
        umi = r1[32:41]
        bead_umi_bc = bead_bc + umi

        raw_umis_per_bead[bead_bc][tag].add(umi)
        raw_umis_per_bead_umi[bead_umi_bc][tag] += 1

    log.debug(f"Beads not in whitelist: {false_beads}")
    log.debug(f"IDs w/o constant sequence: {no_const}")
    log.debug(f"IDs w/o correct ID structure: {no_id_struct}")

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



@click.command()
@click.argument("fastq-r1", type=click.Path(exists=True))
@click.argument("fastq-r2", type=click.Path(exists=True))
@click.option(
    "--puck-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path containing BeadBarcodes.txt and BeadLocations.txt",
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path for output",
    required=True,
)
@click.option(
    "--tag-sequence",
    default="WSBWSDWSHWSVWSBWSWSHWSVWSBWSDWSH",
    help="Format for the tag sequence",
)
@click.option(
    "--constant-sequence",
    default="TCGAGAGATCTACGGGTGGCA",
    help="Constant sequence that should follow the tag",
)
@click.option("--debug", is_flag=True, help="Turn on debug logging")
@click.option(
    "--percentile",
    type=float,
    default=95.0,
    help="Percentile for scaling plots",
    show_default=True,
)
@click.option(
    "--tag-mismatch",
    type=int,
    default=5,
    help="Number of mismatches to allow in tag",
    show_default=True,
)
@click.option(
    "--const-mismatch",
    type=int,
    default=5,
    help="Number of mismatches to allow in constant sequence",
    show_default=True,
)

def main(
    fastq_r1,
    fastq_r2,
    puck_dir,
    output_dir,
    tag_sequence,
    constant_sequence,
    debug=False,
    percentile=95.0,
    tag_mismatch=5,
    const_mismatch=5,
):
    """
    This script generates some plots for mapping barcoded reads.

    Reads sequences from FASTQ_R1 and FASTQ_R2. Assumes that the first read
    contains a 15bp barcode split across two locations, along with an 8bp UMI.
    The second read is assumed to have TAG_SEQUENCE in bases 20-40.
    """
    create_logger(debug, dryrun=False)

    output_dir = Path(output_dir)
    output_pdf = output_dir / "plots.pdf"

    log.info(f"Saving output to {output_dir}")

    log.debug(f"Reading from {fastq_r1}")
    with gzip.open(fastq_r1, "rt") as fh:
        r1_reads = [line.strip() for line in itertools.islice(fh, 1, None, 4)]

    log.debug(f"Reading from {fastq_r2}")
    with gzip.open(fastq_r2, "rt") as fh:
        r2_reads = [line.strip() for line in itertools.islice(fh, 1, None, 4)]

    assert len(r1_reads) == len(r2_reads), "read different number of reads"
    log.info(f"Total of {len(r1_reads)} reads")

    bead_barcodes, xy = get_barcodes(Path(puck_dir))

    constant_sequence_hset = slideseq.bead_matching.hamming_set(
        slideseq.bead_matching.initial_h_set(constant_sequence), d=const_mismatch, include_N=False
    )

    barcode_codes = [DEGENERATE_BASE_DICT[b] for b in tag_sequence]

    # get unique barcodes
    seq_barcodes = sorted({r1[:8] + r1[26:32] for r1 in r1_reads})
    # remove poly-T sequence if present
    seq_barcodes = [seq for seq in seq_barcodes if set(seq) != {"T"}]

    log.info(f"{len(seq_barcodes)} unique seq barcodes")

    log.debug("calculating bead network")
    degen_bead_barcodes, bead_xy, barcode_mapping = bead_network(
        bead_barcodes, seq_barcodes, xy
    )

    barcode_cutoff = len(tag_sequence) - tag_mismatch
    raw_umis_per_bead, raw_umis_per_bead_umi = match_tags(
        r1_reads, r2_reads, barcode_mapping, barcode_codes, barcode_cutoff, constant_sequence_hset, constant_sequence
    )

    slideseq.bead_matching.write_barcode_mapping(
        barcode_mapping, bead_xy, output_dir / "barcode_matching.txt.gz"
    )

    slideseq.bead_matching.write_barcode_xy(
        degen_bead_barcodes, bead_xy, output_dir / "barcode_coordinates.txt.gz"
    )

    filtered_barcodes = [bc for bc in degen_bead_barcodes if bc in raw_umis_per_bead]
    bead_xy_a = np.vstack([bead_xy[dbc] for dbc in filtered_barcodes])

    log.info("Writing output files")
    beads = sorted(raw_umis_per_bead)
    beads_umi = sorted(raw_umis_per_bead_umi)
    raw_tags = sorted({t for b in beads for t in raw_umis_per_bead[b]})

    with gzip.open(output_dir / "beads.txt.gz", "wt") as out:
        print("\n".join(beads), file=out)

    with gzip.open(output_dir / "beads_umi.txt.gz", "wt") as out:
        print("\n".join(beads_umi), file=out)

    with gzip.open(output_dir / "raw_tags.txt.gz", "wt") as out:
        print("\n".join(raw_tags), file=out)

    log.debug("Writing raw umi matrix")
    write_matrix(
        raw_umis_per_bead, beads, raw_tags, output_dir / "raw_umi_matrix.mtx.gz"
    )

    log.debug("Writing umi matrix umi")
    write_matrix(raw_umis_per_bead_umi, beads_umi, raw_tags, output_dir / "raw_umi_matrix_umi.mtx.gz")

    pdf_pages = PdfPages(output_pdf)

    log.info("Making plots")

    umi_dist = [sum(raw_umis_per_bead[bc].values()) for bc in filtered_barcodes]

    spatial_plot(
        bead_xy_a,
        umi_dist,
        "UMIs per bead",
        pdf_pages,
        pct=percentile,
    )

    spatial_plot(
        bead_xy_a,
        np.log10(umi_dist),
        "log10 UMIs per bead",
        pdf_pages,
        pct=percentile,
    )

    pdf_pages.close()
    log.info("Done!")


if __name__ == "__main__":
    main()