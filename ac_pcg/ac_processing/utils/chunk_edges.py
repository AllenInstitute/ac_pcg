import dataclasses

import fastremap
import numpy

from ac_pcg.utils import label_chunk


@dataclasses.dataclass
class ChunkEdgeComponentResult:
    in_chunk_edges: numpy.ndarray
    between_chunk_edges: numpy.ndarray
    chunk_components: numpy.ndarray


@dataclasses.dataclass
class ChunkEdgeResult:
    in_chunk_edges: numpy.ndarray
    between_chunk_edges: numpy.ndarray


def filter_edges_by_chunk(edge_array, query_chunk, labeler):
    segid_bits = labeler.segid_bits
    query_layer_chunk_idx = label_chunk(labeler, query_chunk)

    layer_chunk_edges = edge_array >> segid_bits
    query_chunk_edges_mask = (layer_chunk_edges == query_layer_chunk_idx)
    in_chunk_edges_mask = (
        query_chunk_edges_mask[:, 0] & query_chunk_edges_mask[:, 1]
    )
    between_chunk_edges_mask = (
        query_chunk_edges_mask[:, 0] ^ query_chunk_edges_mask[:, 1]
    )

    in_chunk_edges = edge_array[in_chunk_edges_mask]
    between_chunk_edges = edge_array[between_chunk_edges_mask]

    return ChunkEdgeResult(
        in_chunk_edges=in_chunk_edges,
        between_chunk_edges=between_chunk_edges
    )


def filter_edge_arr_by_vtx(edge_array, vtx):
    return edge_array[
        numpy.all(edge_array != vtx, axis=1)
    ]


def chunk_edges_components_from_skeleton(skel, query_chunk, labeler):
    seg_edges = skel.segments[skel.edges]

    reduced_seg_edges = fastremap.unique(
        numpy.sort(
            seg_edges[
                (seg_edges[:, 0] != seg_edges[:, 1])
            ], axis=1
        ),
        axis=0)

    # NOTE -- there are zeros in components and edges -- filter from full edge list here
    reduced_seg_edges = filter_edge_arr_by_vtx(reduced_seg_edges, 0)

    segid_bits = labeler.segid_bits

    # calculate layer and chunk label for edges
    layer_chunk_edges = reduced_seg_edges >> segid_bits

    # calculate chunk label and find occurences
    query_layer_chunk_idx = label_chunk(labeler, query_chunk)
    query_chunk_edges_mask = (layer_chunk_edges == query_layer_chunk_idx)

    in_chunk_edges_mask = (
        query_chunk_edges_mask[:, 0] & query_chunk_edges_mask[:, 1]
    )
    between_chunk_edges_mask = (
        query_chunk_edges_mask[:, 0] ^ query_chunk_edges_mask[:, 1]
    )

    in_chunk_edges = reduced_seg_edges[in_chunk_edges_mask]
    between_chunk_edges = reduced_seg_edges[between_chunk_edges_mask]

    unique_components = fastremap.unique(
        numpy.concatenate(
            (in_chunk_edges, between_chunk_edges),
            dtype=numpy.int64)
    )
    chunk_components = (unique_components if unique_components.size else None)
    # if unique_components.size:
    #     chunk_components = numpy.concatenate(([unique_components.size], unique_components), dtype=numpy.int64)
    # else:
    #     chunk_components = None

    return ChunkEdgeComponentResult(
        in_chunk_edges=in_chunk_edges,
        between_chunk_edges=between_chunk_edges,
        chunk_components=chunk_components
    )
