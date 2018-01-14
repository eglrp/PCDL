import pyflann
import numpy as np

def fetch_data(model,index,):

    points=[]
    points_num,covar_size,graph_size=2048,16,8

    flann = pyflann.FLANN()
    flann.build_index(points, algorithm='kdtree_simple', leaf_max_size=15)
    indices = np.empty([points_num, graph_size])
    for pt_i, pt in enumerate(points):
        cur_indices, _ = flann.nn_index(pt, covar_size)                  # 1,covar_size
        cur_indices = np.asarray(cur_indices, dtype=np.int).transpose()  # covar_size,1
        indices[pt_i] = cur_indices[1:graph_size, 0]                     # graph_size