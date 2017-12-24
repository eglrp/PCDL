//
// Created by pal on 17-12-12.
//

#include <Python.h>
#include <vector>
#include <cmath>
#include "batch_file_io.h"
#include "mesh_sample.h"
#include "preprocess.h"
#include <numpy/arrayobject.h>
#include <iostream>

static PyObject*
getModelNum(PyObject* self,PyObject* args)
{
    const char* filename;
    if(!PyArg_ParseTuple(args,"s",&filename))
        return NULL;

    unsigned long model_num;
    readBatchFileHead(filename,model_num);

    return Py_BuildValue("i", static_cast<int>(model_num));
}



static PyObject*
getPointCloud(PyObject* self,PyObject* args)
{
    const char* filename;
    unsigned int index;
    unsigned int point_num;
    if(!PyArg_ParseTuple(args,"sii",&filename,&index,&point_num))
        return NULL;

    //read data
    std::vector<double> vertexes;
    std::vector<unsigned int> faces;
    std::vector<double> areas;
    double total_area;
    unsigned int category;
    readBatchFileData(filename,index,vertexes,faces,areas,total_area,category);

    //sample points
    std::vector<double> point_cloud;
    sampleMeshPoints(vertexes,faces,areas,total_area,point_num,point_cloud);

    return Py_BuildValue("s#i", reinterpret_cast<char *>(point_cloud.begin().base()),point_cloud.size()*sizeof(double),category);
}

static PyObject*
getAugmentedPointCloud(PyObject* self,PyObject* args)
{
    const char* filename;
    unsigned int index;
    unsigned int point_num;
    double noise_level;
    if(!PyArg_ParseTuple(args,"siid",&filename,&index,&point_num,&noise_level))
        return NULL;

    //read data
    std::vector<double> vertexes;
    std::vector<unsigned int> faces;
    std::vector<double> areas;
    double total_area;
    unsigned int category;
    readBatchFileData(filename,index,vertexes,faces,areas,total_area,category);

    //sample points
    std::vector<double> point_cloud;
    sampleMeshPoints(vertexes,faces,areas,total_area,point_num,point_cloud);

    //normalize
    normalize(point_cloud);
    //rotated point cloud
    rotatePointCloud(point_cloud);
    //add noise
    addNoise(point_cloud,noise_level);
    //normalize
    normalize(point_cloud);

    return Py_BuildValue("s#i", reinterpret_cast<char *>(point_cloud.begin().base()),point_cloud.size()*sizeof(double),category);
}

static PyObject*
getAugmentedPointCloudWithDiff(PyObject* self,PyObject* args)
{
    const char* filename;
    unsigned int index;
    unsigned int point_num;
    double noise_level;
    if(!PyArg_ParseTuple(args,"siid",&filename,&index,&point_num,&noise_level))
        return NULL;

    //read data
    std::vector<double> vertexes;
    std::vector<unsigned int> faces;
    std::vector<double> areas;
    double total_area;
    unsigned int category;
    readBatchFileData(filename,index,vertexes,faces,areas,total_area,category);

    //sample points
    std::vector<double> point_cloud;
    sampleMeshPoints(vertexes,faces,areas,total_area,point_num,point_cloud);

    //normalize
    normalize(point_cloud);
    //rotated point cloud
    rotatePointCloud(point_cloud);
    //add noise
    addNoise(point_cloud,noise_level);
    //normalize
    normalize(point_cloud);

    std::vector<double> dists;
    computeDists(point_cloud,dists);

    return Py_BuildValue("s#is#", reinterpret_cast<char *>(point_cloud.begin().base()),point_cloud.size()*sizeof(double),category,
                         reinterpret_cast<char *>(dists.begin().base()),dists.size()*sizeof(double));
}

static PyObject*
getPointInterval(PyObject* self,PyObject* args)
{
    PyArrayObject* pts;
    double thresh;
    if(!PyArg_ParseTuple(args,"Od",&pts,&thresh)) return NULL;

    if(pts->nd!=4)
    {
        PyErr_SetString(PyExc_RuntimeError,"Dimension must be 4");
        return NULL;
    }
    if(pts->dimensions[3]!=1||pts->dimensions[2]!=3)
    {
        PyErr_SetString(PyExc_RuntimeError,"Input vector must be [?,?,3,1]");
        return NULL;
    }
    if(pts->descr->type_num!=NPY_FLOAT)
    {
        PyErr_SetString(PyExc_RuntimeError,"Must be float64(double) array");
        return NULL;
    }

    int batch_size=pts->dimensions[0];
    int pt_num=pts->dimensions[1];
    npy_intp dims[3]={batch_size,pt_num,pt_num};
    PyArrayObject* dist_array=reinterpret_cast<PyArrayObject*>(PyArray_New(&PyArray_Type,
                                                                           3,
                                                                           dims,
                                                                           NPY_FLOAT,
                                                                           NULL,
                                                                           NULL,
                                                                           sizeof(float),
                                                                           NULL,
                                                                           NULL));

    int _ps0=pts->strides[0],_ps1=pts->strides[1],_ps2=pts->strides[2];
    int _ds0=dist_array->strides[0],_ds1=dist_array->strides[1],_ds2=dist_array->strides[2];

    for(int i=0;i<batch_size;i++)
    {
        for(int j=0;j<pt_num;j++)
        {
            float x1=*reinterpret_cast<float*>(pts->data+i*_ps0+j*_ps1);
            float y1=*reinterpret_cast<float*>(pts->data+i*_ps0+j*_ps1+1*_ps2);
            float z1=*reinterpret_cast<float*>(pts->data+i*_ps0+j*_ps1+2*_ps2);
            for(int k=0;k<pt_num;k++)
            {
                float x2=*reinterpret_cast<float*>(pts->data+i*_ps0+k*_ps1);
                float y2=*reinterpret_cast<float*>(pts->data+i*_ps0+k*_ps1+1*_ps2);
                float z2=*reinterpret_cast<float*>(pts->data+i*_ps0+k*_ps1+2*_ps2);

                float dist=std::sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
                float weight=1.f-dist/thresh;
                *reinterpret_cast<float*>(dist_array->data+i*_ds0+j*_ds1+k*_ds2)=weight<0.0?0.f:weight;
            }
        }

    }



    return Py_BuildValue("N",dist_array);
}

static PyObject*
getPointCloudRelativePolarForm(PyObject* self,PyObject* args)
{
    const char* filename;
    unsigned int index;
    unsigned int point_num;
    if(!PyArg_ParseTuple(args,"sii",&filename,&index,&point_num))
        return NULL;

    //read data
    std::vector<double> vertexes;
    std::vector<unsigned int> faces;
    std::vector<double> areas;
    double total_area;
    unsigned int category;
    readBatchFileData(filename,index,vertexes,faces,areas,total_area,category);

    //sample points
    std::vector<double> point_cloud;
    sampleMeshPoints(vertexes,faces,areas,total_area,point_num,point_cloud);

    //polar form
    std::vector<double> relative_polar_pc;
    transformToRelativePolarForm(point_cloud,relative_polar_pc);

    return Py_BuildValue("s#i", reinterpret_cast<char *>(relative_polar_pc.begin().base()),
                         relative_polar_pc.size()*sizeof(double),category);
}

static PyMethodDef SampleMethods[]={
        {"getModelNum",getModelNum,METH_VARARGS,""},
        {"getPointCloud",getPointCloud,METH_VARARGS,""},
        {"getAugmentedPointCloud",getAugmentedPointCloud,METH_VARARGS,""},
        {"getPointInterval",getPointInterval,METH_VARARGS,""},
        {"getAugmentedPointCloudWithDiff",getAugmentedPointCloudWithDiff,METH_VARARGS,""},
        {"getPointCloudRelativePolarForm",getPointCloudRelativePolarForm,METH_VARARGS,""},
        {NULL,NULL,NULL,NULL}
};


PyMODINIT_FUNC
initPointSample(void)
{
    (void) Py_InitModule("PointSample", SampleMethods);
    import_array()
}

