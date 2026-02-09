import dataclasses

import cloudvolume
import numpy


@dataclasses.dataclass
class ChunkBbox:
    bbox: cloudvolume.Bbox
    chunk_idx: tuple


def iterate_chunk_slice_boxes(arr_shape, chunk_size):
    for chunk_x, x_min in enumerate(range(0, arr_shape[0], chunk_size[0])):
        x_max = min(arr_shape[0], (x_min + chunk_size[0]))
        for chunk_y, y_min in enumerate(range(0, arr_shape[1], chunk_size[1])):
            y_max = min(arr_shape[1], (y_min + chunk_size[1]))
            for chunk_z, z_min in enumerate(range(0, arr_shape[2], chunk_size[2])):
                z_max = min(arr_shape[2], (z_min + chunk_size[2]))
                chunk_contains_bb = cloudvolume.Bbox(
                    (x_min, y_min, z_min),
                    (x_max, y_max, z_max)
                )
                yield ChunkBbox(
                    bbox=chunk_contains_bb,
                    chunk_idx=(chunk_x, chunk_y, chunk_z)
                )


def chunk_idx_to_bbox(chunk_idx, chunk_size, chunked_box_shape):
    chunk_mins = tuple(
        idx * chunk_d for idx, chunk_d in zip(chunk_idx, chunk_size)
    )
    chunk_max = tuple(
        min(box_d, chunk_min + chunk_d) for chunk_min, chunk_d, box_d in zip(
            chunk_mins, chunk_size, chunked_box_shape)
    )
    bbox = cloudvolume.Bbox(chunk_mins, chunk_max)
    return ChunkBbox(
        bbox=bbox,
        chunk_idx=chunk_idx
    )


def get_bbox_chunks(bbox, chunk_size, offset=None):
    if offset is not None:
        raise NotImplementedError
    (chunk_min, chunk_max), (remainder_min, remainder_max) = numpy.divmod(
        numpy.array([bbox.minpt, bbox.maxpt]), chunk_size)
    # chunk_max += remainder_max.astype(bool)

    # TODO could be more clever about dims
    imin, jmin, kmin = chunk_min
    imax, jmax, kmax = chunk_max
    chunks = numpy.mgrid[
        imin:imax + 1:1,
        jmin:jmax + 1:1,
        kmin:kmax + 1:1
    ].reshape(3, -1).T

    return chunks
