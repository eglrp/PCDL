//
// Created by pal on 18-1-14.
//


#include <Python.h>
#include <vector>
#include <cmath>
#include <numpy/arrayobject.h>
#include <iostream>

# define PYASSERT(val,str) if(!val){PyErr_SetString(PyExc_RuntimeError,(str));return NULL;}
# define PYCHECK(val) if(!val) return NULL

void points2VoxelBatch(
        float* batch_points,    // n,k,3
        float* batch_voxels,    // n,s**3
        int split_num,          // s
        int point_num,          // k
        int batch_size,          // n
        int gpu_index
);

static PyObject*
Points2VoxelBatchGPU(PyObject* self,PyObject* args)
{
    PyArrayObject* pts;
    int split_num,gpu_index;
    if(!PyArg_ParseTuple(args,"Oii",&pts,&split_num,&gpu_index)) return NULL;

    if(pts->nd!=3)
    {
        PyErr_SetString(PyExc_RuntimeError,"Dimension must be 3");
        return NULL;
    }
    if(pts->dimensions[2]!=3)
    {
        PyErr_SetString(PyExc_RuntimeError,"Input vector must be [?,?,3]");
        return NULL;
    }
    if(pts->descr->type_num!=NPY_FLOAT)
    {
        PyErr_SetString(PyExc_RuntimeError,"Must be float array");
        return NULL;
    }

    npy_intp batch_num=pts->dimensions[0];
    npy_intp pt_num=pts->dimensions[1];
    npy_intp dims[2]={batch_num,split_num*split_num*split_num};
    PyArrayObject* voxels=reinterpret_cast<PyArrayObject*>(
            PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT,NULL,NULL,sizeof(float),NULL,NULL));

    Py_BEGIN_ALLOW_THREADS
    points2VoxelBatch(reinterpret_cast<float*>(pts->data),
                      reinterpret_cast<float*>(voxels->data),
                      split_num,pt_num,batch_num,gpu_index);
    Py_END_ALLOW_THREADS

    return Py_BuildValue("N",voxels);
}

void points2VoxelColorBatch(
        float* batch_points,    // n,k,6
        float* batch_voxels,    // n,s**3
        int split_num,          // s
        int point_num,          // k
        int batch_size,          // n
        int gpu_index
);


static PyObject*
Points2VoxeColorlBatchGPU(PyObject* self,PyObject* args)
{
    PyArrayObject* pts;
    int split_num,gpu_index;
    if(!PyArg_ParseTuple(args,"Oii",&pts,&split_num,&gpu_index)) return NULL;

    PYASSERT(pts->nd==3,"Dimension must be 3")
    PYASSERT(pts->dimensions[2]==6,"Input vector must be [?,?,3]")
    PYASSERT(pts->descr->type_num==NPY_FLOAT,"Must be float array")

    npy_intp batch_num=pts->dimensions[0];
    npy_intp pt_num=pts->dimensions[1];
    npy_intp dims[3]={batch_num,split_num*split_num*split_num,4};
    PyArrayObject* voxels=reinterpret_cast<PyArrayObject*>(
            PyArray_New(&PyArray_Type, 3, dims, NPY_FLOAT,NULL,NULL,sizeof(float),NULL,NULL));

    Py_BEGIN_ALLOW_THREADS
    points2VoxelColorBatch(reinterpret_cast<float*>(pts->data),
                           reinterpret_cast<float*>(voxels->data),
                           split_num,pt_num,batch_num,gpu_index);
    Py_END_ALLOW_THREADS

    return Py_BuildValue("N",voxels);
}

void computeCovarsBatch(
        float* batch_points,    // n,k,3
        int* batch_idxs,        // n,k,t
        float* batch_covars,    // n,k,9
        int nn_size,            // t
        int point_num,          // k
        int batch_size,         // n
        int gpu_index
);


static PyObject*
ComputeCovars(PyObject* self,PyObject* args)
{
    PyArrayObject* pts;
    PyArrayObject* nidxs;
    int nn_size,gpu_index;
    if(!PyArg_ParseTuple(args,"OOii",&pts,&nidxs,&nn_size,&gpu_index)) return NULL;

    if(pts->nd!=3)
    {
        PyErr_SetString(PyExc_RuntimeError,"Dimension must be 3");
        return NULL;
    }
    if(pts->dimensions[2]!=3)
    {
        PyErr_SetString(PyExc_RuntimeError,"Input vector must be [?,?,3]");
        return NULL;
    }
    if(pts->descr->type_num!=NPY_FLOAT)
    {
        PyErr_SetString(PyExc_RuntimeError,"Must be float array");
        return NULL;
    }

    if(nidxs->nd!=3)
    {
        PyErr_SetString(PyExc_RuntimeError,"Dimension must be 3");
        return NULL;
    }
    if(nidxs->dimensions[2]!=nn_size)
    {
        PyErr_SetString(PyExc_RuntimeError,"Input vector must be [?,?,nn_size]");
        return NULL;
    }
    if(nidxs->descr->type_num!=NPY_INT)
    {
        PyErr_SetString(PyExc_RuntimeError,"Must be int array");
        return NULL;
    }

    npy_intp batch_num=pts->dimensions[0];
    npy_intp pt_num=pts->dimensions[1];
    npy_intp dims[3]={batch_num,pt_num,9};
    PyArrayObject* covars=reinterpret_cast<PyArrayObject*>(
            PyArray_New(&PyArray_Type, 3, dims, NPY_FLOAT,NULL,NULL,sizeof(float),NULL,NULL));

    Py_BEGIN_ALLOW_THREADS
            computeCovarsBatch(reinterpret_cast<float*>(pts->data),
                               reinterpret_cast<int*>(nidxs->data),
                               reinterpret_cast<float*>(covars->data),
                               nn_size,pt_num,batch_num,gpu_index);
    Py_END_ALLOW_THREADS

    return Py_BuildValue("N",covars);
}

std::vector<std::vector<int> > randomSample(
        float* points,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float block_size,
        float maxx,
        float maxy,
        float min_ratio,
        int split_num,
        float& rot_angle_out
);
std::vector<std::vector<int> >
randomSampleGPU(
        float* points,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float block_size,
        float maxx,
        float maxy,
        float min_ratio,
        int split_num,
        float& rot_angle_out,
        int gpu_index
);

static PyObject*
UniformSampleBlock(PyObject* self,PyObject* args)
{
    PyArrayObject* pts;
    PyArrayObject* lbs;
    float sample_stride;
    float block_size;
    float maxx,maxy;
    float ratio;
    int split_num;
    if(!PyArg_ParseTuple(args,"OOfffiff",&pts,&lbs,&sample_stride,&block_size,&ratio,&split_num,&maxx,&maxy)) return NULL;
    PYASSERT(pts->nd==2,"Dimension must larger than 3")
    PYASSERT(pts->dimensions[1]>3,"dims[1] need >=3")
    PYASSERT(pts->descr->type_num==NPY_FLOAT,"Must be float array")
    PYASSERT(lbs->nd==2,"Dimension must be 2")
    PYASSERT(lbs->dimensions[1]==1,"Input vector must be [?,1]")

    float* p_pts=reinterpret_cast<float*>(pts->data);
    int* p_lbs=reinterpret_cast<int*>(lbs->data);
    float out_angle;


    npy_intp pt_num=pts->dimensions[0];
    npy_intp pt_stride=pts->dimensions[1];
    //printf("pt_num %d pt_stride %d\n",pt_num,pt_stride);
    //printf("coord %f %f %f %f %f %f\n",p_pts[0],p_pts[1],p_pts[2],p_pts[3],p_pts[4],p_pts[5]);
    std::vector<std::vector<int> > block_idxs;
    Py_BEGIN_ALLOW_THREADS;
//    block_idxs=randomSample(reinterpret_cast<float*>(pts->data),pt_num,pt_stride,sample_stride,
//                            block_size,maxx,maxy,ratio,split_num,out_angle);
    block_idxs=randomSampleGPU(reinterpret_cast<float*>(pts->data),pt_num,pt_stride,sample_stride,
                            block_size,maxx,maxy,ratio,split_num,out_angle,0);
    Py_END_ALLOW_THREADS;

    PyObject* block_points_list = reinterpret_cast<PyObject*>(PyList_New(0));
    PYASSERT(block_points_list!=NULL,"block_points_list error");
    PyObject* block_labels_list = reinterpret_cast<PyObject*>(PyList_New(0));
    PYASSERT(block_labels_list!=NULL,"block_points_list error");

    // compute rotation matrix
    float m00=cos(out_angle),m01=-sin(out_angle);
    float m10=-m01,m11=m00;

    //printf("block num %d\n",int(block_idxs.size()));
    for(int i=0;i<block_idxs.size();i++)
    {
        //printf("block %d size %d\n",i,int(block_idxs[i].size()));
        if(block_idxs[i].size()<1024)
            continue;

        npy_intp dims[2]={block_idxs[i].size(),pt_stride};
        PyArrayObject* block_pts=reinterpret_cast<PyArrayObject*>(
                PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT,NULL,NULL,sizeof(float),NULL,NULL));
        float* p_bpts=reinterpret_cast<float*>(block_pts->data);

        npy_intp labels_dims[2]={block_idxs[i].size(),1};
        PyArrayObject* block_lbs=reinterpret_cast<PyArrayObject*>(
                PyArray_New(&PyArray_Type, 2, labels_dims, NPY_INT,NULL,NULL,sizeof(int),NULL,NULL));
        int* p_blbs=reinterpret_cast<int*>(block_lbs->data);

        for(int j=0;j<block_idxs[i].size();j++)
        {
            memcpy(&(p_bpts[j*pt_stride]),
                   &(p_pts[block_idxs[i][j]*pt_stride]),
                   pt_stride*sizeof(float));

            // rotate back
            float x0=p_bpts[j*pt_stride];
            float y0=p_bpts[j*pt_stride+1];
            p_bpts[j*pt_stride]=x0*m00+y0*m01;
            p_bpts[j*pt_stride+1]=x0*m10+y0*m11;

            memcpy(&(p_blbs[j]),
                   &(p_lbs[block_idxs[i][j]]),
                    sizeof(int));
        }

        PYASSERT(PyList_Append(block_points_list,reinterpret_cast<PyObject*>(block_pts))==0,"append error");
        PYASSERT(PyList_Append(block_labels_list,reinterpret_cast<PyObject*>(block_lbs))==0,"append error");

        Py_DECREF(block_lbs);
        Py_DECREF(block_pts);
    }

    return Py_BuildValue("NN", block_points_list, block_labels_list);
}

void gridDownSample(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& downsample_indices
);


static PyObject*
GridDownSample(PyObject* self,PyObject* args)
{
    PyArrayObject* pts;
    PyArrayObject* lbs;
    float sample_stride;
    if(!PyArg_ParseTuple(args,"OOf",&pts,&lbs,&sample_stride)) return NULL;
    PYASSERT(pts->nd==2,"Dimension must larger than 3")
    PYASSERT(pts->dimensions[1]>3,"dims[1] need >=3")
    PYASSERT(pts->descr->type_num==NPY_FLOAT,"Must be float array")
    PYASSERT(lbs->nd==2,"Dimension must be 2")
    PYASSERT(lbs->dimensions[1]==1,"Input vector must be [?,1]")

    float* p_pts=reinterpret_cast<float*>(pts->data);
    int* p_lbs=reinterpret_cast<int*>(lbs->data);

    npy_intp pt_num=pts->dimensions[0];
    npy_intp pt_stride=pts->dimensions[1];;
    std::vector<int> downsample_idxs;

    Py_BEGIN_ALLOW_THREADS
    gridDownSample(p_pts,pt_num,pt_stride,sample_stride,downsample_idxs);
    Py_END_ALLOW_THREADS

    npy_intp dims[2]={downsample_idxs.size(),pt_stride};
    PyArrayObject* downsample_pts=reinterpret_cast<PyArrayObject*>(
            PyArray_New(&PyArray_Type, 2, dims, NPY_FLOAT,NULL,NULL,sizeof(float),NULL,NULL));
    float* p_dspts=reinterpret_cast<float*>(downsample_pts->data);

    npy_intp label_dims[2]={downsample_idxs.size(),1};
    PyArrayObject* downsample_lbs=reinterpret_cast<PyArrayObject*>(
            PyArray_New(&PyArray_Type, 2, label_dims, NPY_INT, NULL,NULL,sizeof(int),NULL,NULL));
    int* p_dslbs=reinterpret_cast<int*>(downsample_lbs->data);

    for(int i=0;i<downsample_idxs.size();i++)
    {
        memcpy(
                &(p_dspts[i*pt_stride]),
                &(p_pts[downsample_idxs[i]*pt_stride]),
                pt_stride*sizeof(float)
        );
        memcpy(
                &(p_dslbs[i]),
                &(p_lbs[downsample_idxs[i]]),
                sizeof(int)
        );
    }

    return Py_BuildValue("NN", downsample_pts, downsample_lbs);
}

static PyMethodDef PointsUtilMethods[]={
        {"Points2VoxelBatchGPU",Points2VoxelBatchGPU,METH_VARARGS,""},
        {"Points2VoxeColorlBatchGPU",Points2VoxeColorlBatchGPU,METH_VARARGS,""},
        {"ComputeCovars",ComputeCovars,METH_VARARGS,""},
        {"UniformSampleBlock",UniformSampleBlock,METH_VARARGS,""},
        {"GridDownSample",GridDownSample,METH_VARARGS,""},
        {NULL,NULL,NULL,NULL}
};


PyMODINIT_FUNC
initPointsUtil(void)
{
    (void) Py_InitModule("PointsUtil", PointsUtilMethods);
    import_array()
}