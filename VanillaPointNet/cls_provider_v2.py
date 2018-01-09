from provider import *
import h5py
import numpy as np
import functools

def rotate(pts,rotation_angle=None):
    if rotation_angle is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]], dtype=np.float32)
    rotated_pts = np.dot(pts, rotation_matrix)
    return rotated_pts

def exchange_dims_zy(pcs):
    #pcs [n,k,3]
    exchanged_data = np.empty(pcs.shape, dtype=np.float32)

    exchanged_data[:,:,0]=pcs[:,:,0]
    exchanged_data[:,:,1]=pcs[:,:,2]
    exchanged_data[:,:,2]=pcs[:,:,1]
    return exchanged_data

def read_all_data(file_list):
    points,normals,nidxs,mdists,labels=[],[],[],[],[]
    for fn in file_list:
        f=h5py.File(fn,'r')
        points.append(f['point'][:])
        normals.append(f['normal'][:])
        nidxs.append(f['nidx'][:])
        mdists.append(f['mdist'][:])
        labels.append(f['label'][:])
        f.close()

    points=np.concatenate(points,axis=0)
    normals=np.concatenate(normals,axis=0)
    nidxs=np.concatenate(nidxs,axis=0)
    mdists=np.concatenate(mdists,axis=0)
    labels=np.concatenate(labels,axis=0)

    points=exchange_dims_zy(points)
    normals=exchange_dims_zy(normals)

    return points,normals,nidxs,mdists,labels

def fetch_data(model,index,points,labels,normals=None,nidxs=None,mdists=None,noise_level=1e-3):
    # transform
    if model=='train':
        batch_points=points[index]
        # batch_points+=np.random.normal(0,noise_level,batch_points.shape)
        batch_out_points=batch_points+normals[index]*np.expand_dims(mdists[index],axis=1)
        batch_points=np.concatenate([batch_points,batch_out_points],axis=0)
        batch_points=rotate(batch_points)

        return batch_points,labels[index],nidxs[index]
    else:
        return points[index],labels[index]

def fetch_batch(model,batch_raw):
    if model=='train':
        batch_points,batch_nidxs,batch_labels=[],[],[]
        for bi,b in enumerate(batch_raw):
            batch_points.append(np.expand_dims(b[0],axis=0))
            batch_labels.append(np.expand_dims(b[1],axis=0))
            nidxs=b[2]
            nidxs=np.expand_dims(nidxs,axis=2)                  # k t 1
            batch_idxs=np.full_like(nidxs,bi)                   # k t 1
            nidxs=np.concatenate([batch_idxs,nidxs],axis=2)     # k t 2
            batch_nidxs.append(np.expand_dims(nidxs,axis=0))    # 1 k t 2
        return np.concatenate(batch_points,axis=0),np.concatenate(batch_labels,axis=0),np.concatenate(batch_nidxs,axis=0)
    else:
        batch_points,batch_labels=[],[]
        for b in batch_raw:
            batch_points.append(np.expand_dims(b[0],axis=0))
            batch_labels.append(np.expand_dims(b[1],axis=0))
        return np.concatenate(batch_points,axis=0),np.concatenate(batch_labels,axis=0)

if __name__=="__main__":
    train_file_list=['modelnet40_util/ply_data_train{}.h5'.format(i) for i in range(0,5)]
    test_file_list=['modelnet40_util/ply_data_test{}.h5'.format(i) for i in range(1,2)]

    train_points,train_normals,train_nidxs,train_mdists,train_labels=read_all_data(test_file_list)
    train_fetch_data=functools.partial(fetch_data,points=train_points,labels=train_labels,
                      normals=train_normals,nidxs=train_nidxs,mdists=train_mdists)
    input_list=[(i,) for i in range(train_points.shape[0])]
    provider=Provider(input_list,16,train_fetch_data,'train',batch_fn=fetch_batch)

    try:
        for batch_points,batch_labels,batch_nidxs in provider:
            points=batch_points[0]
            nidxs=batch_nidxs[0]

            with open('model.txt', 'w') as f:
                for pt in points[:2048]:
                    f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            with open('normal.txt', 'w') as f:
                for pt in points[2048:]:
                    f.write('{} {} {}\n'.format(pt[0], pt[1], pt[2]))

            for i in range(30):
                with open('local{}.txt'.format(i), 'w') as f:
                    colors=np.random.randint(0,256,[3])
                    opt=points[i,:]
                    npt=points[i+2048,:]
                    f.write('{} {} {} {} {} {}\n'.format(opt[0], opt[1], opt[2], 128,128,128))
                    f.write('{} {} {} {} {} {}\n'.format(npt[0], npt[1], npt[2], 255, 255, 255))
                    for idx in nidxs[i,:]:
                        f.write('{} {} {} {} {} {}\n'.format(points[idx][0], points[idx][1], points[idx][2], colors[0], colors[1], colors[2]))
            break
    finally:
        provider.close()
