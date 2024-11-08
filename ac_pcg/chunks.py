import cloudvolume
import dataclasses


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