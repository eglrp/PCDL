import h5py
import pyflann
import numpy as np

def save_h5(filename,points,normals,indices,eigvals,mdists,labels):
    h5_fout = h5py.File(filename)
    h5_fout.create_dataset(
            'point', data=points,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'label', data=labels,
            compression='gzip', compression_opts=1,
            dtype='uint8')
    h5_fout.create_dataset(
            'normal', data=normals,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'eigval', data=eigvals,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.create_dataset(
            'nidx', data=indices,
            compression='gzip', compression_opts=4,
            dtype='uint32')
    h5_fout.create_dataset(
            'mdist', data=mdists,
            compression='gzip', compression_opts=4,
            dtype='float32')
    h5_fout.close()

def compute_normal(points,k=5):
    assert k>=5
    flann=pyflann.FLANN()
    flann.build_index(points,algorithm='kdtree_simple',leaf_max_size=3)
    normals=[]
    eigvals=[]
    indices=[]
    mdists=[]
    for pt in points:
        nidxs,ndists=flann.nn_index(pt,k+1)
        npts=points[nidxs[0,:],:]
        # compute normal
        mean=np.mean(npts,axis=0,keepdims=True)
        var=(npts-mean).transpose().dot(npts-mean)
        eigval,eigvec=np.linalg.eigh(var)
        normals.append(np.expand_dims(eigvec[0],axis=0))
        eigvals.append(np.expand_dims(eigval,axis=0))
        # only use 5 points
        indices.append(nidxs[:,1:6])
        mdists.append(np.mean(np.sqrt(ndists[0,1:6]),axis=0))

    normals=np.concatenate(normals,axis=0)
    eigvals=np.concatenate(eigvals,axis=0)
    indices=np.concatenate(indices,axis=0)
    mdists=np.stack(mdists,axis=0)
    # flip normals
    masks=np.sum(normals*points,axis=1)<0
    normals[masks]=-normals[masks]

    return normals,eigvals,indices,mdists


if __name__=="__main__":
    # filename='ply_data_train0'
    filename_list=[
        # 'ply_data_train0',
        # 'ply_data_train1',
        # 'ply_data_train2',
        # 'ply_data_train3',
        # 'ply_data_train4',
        # 'ply_data_test0',
        'ply_data_test1',
    ]
    for filename in filename_list:
        f=h5py.File('../data/ModelNet40PointNetSampleVersion/modelnet40_ply_hdf5_2048/{}.h5'.format(filename),'r')
        data=f['data'][:]
        label=f['label'][:]
        f.close()

        print data.shape

        indices,normals,eigvals,mdists=[],[],[],[]
        for pts_i,pts in enumerate(data):
            normal,eigval,index,mdist=compute_normal(pts,15)
            # print normal.shape
            # print eigval.shape
            # print index.shape
            # print mdist.shape
            indices.append(np.expand_dims(index,axis=0))
            normals.append(np.expand_dims(normal,axis=0))
            eigvals.append(np.expand_dims(eigval,axis=0))
            mdists.append(np.expand_dims(mdist,axis=0))

            # with open('pts.txt','w') as f:
            #     for pt,nr in zip(pts,normal):
            #         f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],nr[0],nr[1],nr[2]))
            #
            # with open('nrms.txt','w') as f:
            #     npts=normal*1e-1+pts
            #     for pt in npts:
            #         f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            if pts_i%500==0:
                print '{} done'.format(pts_i)


        normals=np.concatenate(normals,axis=0)
        eigvals=np.concatenate(eigvals,axis=0)
        indices=np.concatenate(indices,axis=0)
        mdists=np.concatenate(mdists,axis=0)
        save_h5(filename+'.h5',data,normals,indices,eigvals,mdists,label)