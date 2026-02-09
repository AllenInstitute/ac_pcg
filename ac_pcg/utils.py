import numpy

np = numpy


def label_chunk(labeler, chunk):
    layer_offset = numpy.uint64(64 - labeler.n_bits_for_layer_id)
    x_offset = numpy.uint64(layer_offset - labeler.spatial_bit_count)
    y_offset = numpy.uint64(x_offset - labeler.spatial_bit_count)
    z_offset = numpy.uint64(y_offset - labeler.spatial_bit_count)

    layer = numpy.uint64(labeler.level)
    x, y, z = numpy.uint64(chunk)

    segid_bits = labeler.segid_bits

    return numpy.uint64(
        layer << layer_offset | x << x_offset | y << y_offset | z << z_offset
    ) >> segid_bits


def chunk_edges_from_skeleton(skel, query_chunk, labeler):
    seg_edges = skel.segments[skel.edges]
    reduced_seg_edges = numpy.unique(
        numpy.sort(
            seg_edges[
                (seg_edges[:, 0] != seg_edges[:, 1])
            ], axis=1
        ),
        axis=0)

    segid_bits = labeler.segid_bits

    # calculate layer and chunk label for edges
    layer_chunk_edges = reduced_seg_edges >> segid_bits

    # calculate chunk label and find occurences
    query_layer_chunk_idx = label_chunk(labeler, query_chunk)
    query_chunk_edges_mask = (layer_chunk_edges == query_layer_chunk_idx)

    in_chunk_edges_mask = query_chunk_edges_mask[:, 0] & query_chunk_edges_mask[:, 1]
    between_chunk_edges_mask = query_chunk_edges_mask[:, 0] ^ query_chunk_edges_mask[:, 1]    

    in_chunk_edges = reduced_seg_edges[in_chunk_edges_mask]
    between_chunk_edges = reduced_seg_edges[between_chunk_edges_mask]
    return in_chunk_edges, between_chunk_edges
