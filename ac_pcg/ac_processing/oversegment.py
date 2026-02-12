import fastremap
import kimimaro


def process_oversegment_array(arr, skels, label_func, oversegment_kwargs=None, relabel_zero=False):
    oversegment_kwargs = oversegment_kwargs or {}
    oversegmented_arr, oversegmented_skels = kimimaro.utility.oversegment(
        arr, skels, **oversegment_kwargs)
    # convert labels to uint64 layer/chunk indices
    lbl_map = {
        input_lbl: label_func(input_lbl) # input_lbl: labeler.encode_chunk_seg(chunk_box.chunk_idx, input_lbl)
        for input_lbl in fastremap.unique(
            oversegmented_arr
        )
    }
    if not relabel_zero:
        lbl_map[0] = 0
    relabeled_arr = fastremap.remap(oversegmented_arr, lbl_map)
    for skel in oversegmented_skels:
        skel.segments = fastremap.remap(skel.segments, lbl_map)
    return relabeled_arr, oversegmented_skels
