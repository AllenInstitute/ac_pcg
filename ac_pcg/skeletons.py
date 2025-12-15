import cloudvolume
import numpy

# similar to skeleton.crop, except it returns skeleton object
#   and indices of vertices into original skeleton
Bbox = cloudvolume.Bbox
np = numpy

def crop_skel(skel, bbox, return_indices=False, remove_disconnected_vertices=True):
    skeleton = skel.clone()
    bbox = Bbox.create(bbox)

    if skeleton.empty():
        if return_indices:
            return skeleton, None
        else:
            return skeleton

    nodes_valid_mask = np.array(
      [ bbox.contains(vtx) for vtx in skeleton.vertices ], dtype=bool
    )
    nodes_valid_idx = np.where(nodes_valid_mask)[0]

    # Set invalid vertices to be duplicates
    # so they'll be removed during consolidation
    if nodes_valid_idx.shape[0] == 0:
        if return_indices:
            return cloudvolume.Skeleton(), None
        else:
            return cloudvolume.Skeleton()

    first_node = nodes_valid_idx[0]
    skeleton.vertices[~nodes_valid_mask] = skeleton.vertices[first_node]
  
    edges_valid_mask = np.isin(skeleton.edges, nodes_valid_idx)
    edges_valid_idx = edges_valid_mask[:,0] * edges_valid_mask[:,1] 
    skeleton.edges = skeleton.edges[edges_valid_idx,:]

    if return_indices:
        # crop_indices = numpy.argwhere(edges_valid_idx)
        return skeleton.consolidate(remove_disconnected_vertices=remove_disconnected_vertices), nodes_valid_idx
    else:
        return skeleton.consolidate(remove_disconnected_vertices=remove_disconnected_vertices)


def bboxed_skel(skel, bb):
    new_skel, indices = crop_skel(skel, bb, return_indices=True, remove_disconnected_vertices=False)
    new_skel.vertices -= bb.minpt

    if new_skel.vertices.size:
        return new_skel, indices