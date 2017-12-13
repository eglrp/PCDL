//
// Created by pal on 17-12-12.
//
#include <cassert>
#include "batch_file_io.h"

using namespace std;

void readBatchFileHead(
        const std::string& filename,
        unsigned long& model_num
)
{
    FILE* fp=fopen(filename.c_str(),"r");
    if(!fp) return;
    fread(&model_num,sizeof(unsigned long),1,fp);
    fclose(fp);
}

void readBatchFileData(
        const std::string& file_name,
        unsigned int index,
        std::vector<double>& vertexes,
        std::vector<unsigned int>& faces,
        std::vector<double>& areas,
        double& total_area,
        unsigned int& category_index
)
{

    FILE* fp=fopen(file_name.c_str(),"r");
    if(!fp) return;
    //read head
    unsigned long cur_batch_size;
    fread(&cur_batch_size,sizeof(unsigned long),1,fp);
    vector<unsigned long> offsets(cur_batch_size);
    fread(offsets.begin().base(),sizeof(unsigned long),cur_batch_size,fp);

    assert(cur_batch_size>index);

    fseek(fp,offsets[index],SEEK_SET);
    unsigned long vert_num,face_num;

    fread(&vert_num,sizeof(unsigned long),1,fp);       //read vert_num
    fread(&face_num,sizeof(unsigned long),1,fp);       //read face_num
    fread(&total_area,sizeof(double),1,fp);            //read total area
    fread(&category_index,sizeof(unsigned int),1,fp);  //read category index

    vertexes.resize(3*vert_num);
    fread(vertexes.begin().base(),sizeof(double),3*vert_num,fp);

    faces.resize(3*face_num);
    fread(faces.begin().base(),sizeof(unsigned int),3*face_num,fp);

    areas.resize(face_num);
    fread(areas.begin().base(),sizeof(double),face_num,fp);

    fclose(fp);
}