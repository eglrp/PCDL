//
// Created by pal on 17-12-12.
//

#include "batch_file_io.h"
#include "mesh_sample.h"
#include "preprocess.h"

#include <iostream>
#include <fstream>

using namespace std;


void writePointCloud(
        const string &file_name,
        const vector<double> &point_cloud
)
{
    ofstream fout(file_name);
    if(!fout.is_open()) return;
    fout.setf(ios::fixed);
    fout.precision(6);
    for(size_t i=0;i<point_cloud.size()/3;i++)
    {
        fout<<point_cloud[i*3]<<" ";
        fout<<point_cloud[i*3+1]<<" ";
        fout<<point_cloud[i*3+2]<<"\n";
    }

    fout.flush();
    fout.close();
}

void writePointCloud(
        const string &file_name,
        double* point_cloud,
        int point_cloud_size
)
{
    ofstream fout(file_name);
    if(!fout.is_open()) return;
    fout.setf(ios::fixed);
    fout.precision(6);
    for(size_t i=0;i<point_cloud_size/3;i++)
    {
        fout<<point_cloud[i*3]<<" ";
        fout<<point_cloud[i*3+1]<<" ";
        fout<<point_cloud[i*3+2]<<"\n";
    }

    fout.flush();
    fout.close();
}

void writeOFF(
        const string &file_name,
        const vector<double>& vertexes,
        const vector<unsigned int>& faces
)
{
    ofstream fout(file_name);
    if(!fout.is_open()) return;
    fout<<"OFF"<<endl;
    fout<<vertexes.size()/3<<" "<<faces.size()/3<<" 0\n";
    fout.setf(ios::fixed);
    fout.precision(6);
    for(size_t i=0;i<vertexes.size()/3;i++)
    {
        fout<<vertexes[i*3]<<" ";
        fout<<vertexes[i*3+1]<<" ";
        fout<<vertexes[i*3+2]<<"\n";
    }
    for(size_t i=0;i<faces.size()/3;i++)
    {
        fout<<"3 ";
        fout<<faces[i*3]<<" ";
        fout<<faces[i*3+1]<<" ";
        fout<<faces[i*3+2]<<"\n";
    }

    fout.flush();
    fout.close();
}


int main()
{
    cout<<sizeof(double)<<endl;
    string filename="/home/pal/data/ModelNet40/test0.batch";
    unsigned long model_num;
    readBatchFileHead(filename,model_num);
    srand(time(0));
    unsigned int index=26;

    vector<double> vertexes;
    vector<unsigned int> faces;
    vector<double> areas;
    double total_area;
    unsigned int category;
    readBatchFileData(filename,index,vertexes,faces,areas,total_area,category);

    cout<<"category:"<<category<<endl;

    vector<double> point_cloud;
    sampleMeshPoints(vertexes,faces,areas,total_area,10240,point_cloud);

    //normalize
    normalize(point_cloud);
    //rotated point cloud
    rotatePointCloud(point_cloud);
    //add noise
    addNoise(point_cloud,1e-2);
    //normalize
    normalize(point_cloud);

    writePointCloud("test.txt",point_cloud);

    writeOFF("test.off",vertexes,faces);

    return 0;
}
