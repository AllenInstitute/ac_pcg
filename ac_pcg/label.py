# modified from cloudvolume
import dataclasses

import numpy

uint64 = numpy.uint64


def encode_label(layer, x, y, z, segid, n_bits_for_layer_id=8, spatial_bit_count=8):
    """
    Create a graphene label from the specified values.

    Another way to use this:

    glabel = GrapheneLabel(2,1,1,1,777)
    meta.encode_label(*glabel)
    """
    spatial_bit_count = uint64(spatial_bit_count)

    layer_offset = uint64(64 - n_bits_for_layer_id)
    x_offset = uint64(layer_offset - spatial_bit_count)
    y_offset = uint64(x_offset - spatial_bit_count)
    z_offset = uint64(y_offset - spatial_bit_count)

    if not (
      x < 2 ** spatial_bit_count and y < 2 ** spatial_bit_count and z < 2 ** spatial_bit_count
    ):
      raise ValueError(
        "Chunk coordinate is out of range for "
        "this graph on layer %d with %d bits/dim. "
        "[%d, %d, %d]; max = %d."
        % (layer, spatial_bit_count, x, y, z, 2 ** spatial_bit_count)
      )

    segid_bits = uint64(64 - n_bits_for_layer_id - 3 * spatial_bit_count)

    if segid >= 2 ** segid_bits:
      raise ValueError(
        "segid {} provided is out of range. It must be less than {}".format(
          segid, 2 ** segid_bits
      ))

    layer = uint64(layer)
    x, y, z = uint64(x), uint64(y), uint64(z)
    segid = uint64(segid)

    return uint64(
      layer << layer_offset | x << x_offset | y << y_offset | z << z_offset | segid
    )


def decode_layer_id(label, n_bits_for_layer_id=8):
    return uint64(label) >> uint64(64 - n_bits_for_layer_id)


def decode_segid(label, n_bits_for_layer_id=8, spatial_bit_count=8):
    label = uint64(label)
    level = decode_layer_id(label)
    segid_bits = uint64(64 - n_bits_for_layer_id - 3 * spatial_bit_count)

    mask = uint64(2 ** segid_bits) - uint64(1)
    
    return label & mask

def decode_chunk_position(label, n_bits_for_layer_id=8, spatial_bit_count=8):
    """Returns the chunk position as a tuple (X,Y,Z)"""
    label = uint64(label)
    level = decode_layer_id(label)
    spatial_bit_count = uint64(spatial_bit_count)
    label = label & uint64(0x00ffffffffffffff)
    masks = spatial_bit_masks(level)
    segid_bits = uint64(64 - n_bits_for_layer_id - 3 * spatial_bit_count)

    x = (label & masks[0]) >> uint64(segid_bits + 2 * spatial_bit_count)
    y = (label & masks[1]) >> uint64(segid_bits + 1 * spatial_bit_count)
    z = (label & masks[2]) >> uint64(segid_bits + 0 * spatial_bit_count)

    return (x,y,z)

def spatial_bit_masks(level, n_bits_for_layer_id=8, spatial_bit_count=8):
    mask = uint64(2 ** spatial_bit_count) - uint64(1)
    segid_bits = 64 - n_bits_for_layer_id - 3 * spatial_bit_count

    return [
      mask << uint64(segid_bits + 2 * spatial_bit_count),
      mask << uint64(segid_bits + 1 * spatial_bit_count),
      mask << uint64(segid_bits + 0 * spatial_bit_count)
    ]

def decode_label(label, n_bits_for_layer_id=8, spatial_bit_count=8):
    level = decode_layer_id(label, n_bits_for_layer_id=8)
    x,y,z = decode_chunk_position(label, n_bits_for_layer_id=8, spatial_bit_count=8)
    segid = decode_segid(label, n_bits_for_layer_id=8, spatial_bit_count=8)
    return (level, x, y, z, segid)


@dataclasses.dataclass
class ChunkLabeler:
    level: int = 1
    n_bits_for_layer_id: int = 8
    spatial_bit_count: int = 8

    def encode_chunk_seg(self, chunk_vec, seg_id):
        return encode_label(
            self.level, chunk_vec[0], chunk_vec[1], chunk_vec[2], seg_id,
            n_bits_for_layer_id=self.n_bits_for_layer_id,
            spatial_bit_count=self.spatial_bit_count
        )
        
    def decode_label(self, label):
        return decode_label(
            label,
            n_bits_for_layer_id=self.n_bits_for_layer_id,
            spatial_bit_count=self.spatial_bit_count
        )
