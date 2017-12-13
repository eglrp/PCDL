//
// Created by pal on 17-12-12.
//

#include <Python.h>
#include <vector>
#include "batch_file_io.h"
#include "mesh_sample.h"


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

    FILE* fp=fopen("test_buffer_cpp.txt","wb");
    fwrite(point_cloud.begin().base(),sizeof(double),point_cloud.size(),fp);
    fclose(fp);

    return Py_BuildValue("s#i", reinterpret_cast<char *>(point_cloud.begin().base()),point_cloud.size()*sizeof(double),category);
}

static PyMethodDef SampleMethods[]={
        {"getModelNum",getModelNum,METH_VARARGS,"get the model number in the given bat file"},
        {"getPointCloud",getPointCloud,METH_VARARGS,"sample n points of the ith model in the given file"},
        {NULL,NULL,NULL,NULL}
};


PyMODINIT_FUNC
initpoint_sample(void)
{
    (void) Py_InitModule("point_sample", SampleMethods);
}

