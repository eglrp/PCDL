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
getPointCloudNormal(PyObject* self,PyObject* args)
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
    std::vector<double> normal_cloud;
    sampleMeshPoints(vertexes,faces,areas,total_area,point_num,point_cloud,normal_cloud);

    //convert to numpy array
    npy_intp pt_num=point_cloud.size()/3;
    npy_intp dims[2]={pt_num,3};
    PyArrayObject* pts=reinterpret_cast<PyArrayObject*>(
            PyArray_New(&PyArray_Type,2,dims, NPY_FLOAT,NULL,NULL,sizeof(float),NULL,NULL));
    PyArrayObject* normals=reinterpret_cast<PyArrayObject*>(
            PyArray_New(&PyArray_Type,2,dims, NPY_FLOAT,NULL,NULL,sizeof(float),NULL,NULL));

    npy_int _ps0=pts->strides[0],_ps1=pts->strides[1],
            _ns0=normals->strides[0],_ns1=normals->strides[1];
    for(int i=0;i<pt_num;i++)
    {
        *reinterpret_cast<float*>(pts->data+i*_ps0)=static_cast<float>(point_cloud[i*3]);
        *reinterpret_cast<float*>(pts->data+i*_ps0+1*_ps1)=static_cast<float>(point_cloud[i*3+1]);
        *reinterpret_cast<float*>(pts->data+i*_ps0+2*_ps1)=static_cast<float>(point_cloud[i*3+2]);
        *reinterpret_cast<float*>(normals->data+i*_ns0)=static_cast<float>(normal_cloud[i*3]);
        *reinterpret_cast<flo   at*>(normals->data+i*_ns0+1*_ns1)=static_cast<float>(normal_cloud[i*3+1]);
        *reinterpret_cast<float*>(normals->data+i*_ns0+2*_ns1)=static_cast<float>(normal_cloud[i*3+2]);
    }

    return Py_BuildValue("NNi",pts,normals,category);
}


static PyMethodDef SampleMethods[]={
        {"getModelNum",getModelNum,METH_VARARGS,""},
        {"getPointCloud",getPointCloud,METH_VARARGS,""},
        {"getAugmentedPointCloud",getAugmentedPointCloud,METH_VARARGS,""},
        {"getPointCloudNormal",getPointCloudNormal,METH_VARARGS,""},
        {NULL,NULL,NULL,NULL}
};


PyMODINIT_FUNC
initPointSample(void)
{
    (void) Py_InitModule("PointSample", SampleMethods);
    import_array()
}

