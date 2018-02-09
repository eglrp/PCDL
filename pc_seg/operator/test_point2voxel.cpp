//
// Created by pal on 18-1-14.
//

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <utility>
#include <iostream>

std::vector<float> readTXT(const std::string& fn)
{
    std::ifstream fin;
    fin.open(fn.c_str());
    std::string line;
    std::vector<float> pts;
    while(std::getline(fin,line))
    {
//        std::cout<<line<<std::endl;
        std::stringstream ss(line);
        float x,y,z;
        ss>>x>>y>>z;
        pts.push_back(x+0.5);
        pts.push_back(y+0.5);
        pts.push_back(z);
    }
    fin.close();

    return pts;
}

void writeTXT(const std::string& fn,const std::vector<float>& pts)
{
    std::ofstream fout;
    fout.open(fn.c_str());
    int pt_num=pts.size()/3;
    for(int i=0;i<pt_num;i++)
    {
        fout<<pts[i*3]<<" "<<pts[i*3+1]<<" "<<pts[i*3+2]<<std::endl;
    }
    fout.close();
}

std::vector<float> voxel2points(int split_num,const std::vector<float>& voxels)
{
    float stride=stride=1.f/(split_num-1);
    std::vector<float> pts;
    for(int i=0;i<split_num;i++)
        for(int j=0;j<split_num;j++)
            for(int k=0;k<split_num;k++)
            {
                if(voxels[i*split_num*split_num+j*split_num+k]>1e-5)
                {
                    pts.push_back(i*stride);
                    pts.push_back(j*stride);
                    pts.push_back(k*stride);
                }
            }

    return pts;
}

void points2Voxel(
        float* points,
        float* voxels,
        int split_num,
        int point_num
);
int main()
{
    std::vector<float> pts=readTXT("test.txt");
    writeTXT("result1.txt",pts);
    int split_num=30;
    std::vector<float> voxels(split_num*split_num*split_num);
    points2Voxel(pts.data(),voxels.data(),split_num,pts.size()/3);
    std::vector<float> voxel_pts=voxel2points(split_num,voxels);

    writeTXT("result3.txt",pts);
    writeTXT("result2.txt",voxel_pts);

    return 0;
}
