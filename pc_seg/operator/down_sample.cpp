//
// Created by pal on 18-1-25.
//

#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>
#include <iostream>
#include <sstream>

void gridDownSampleIdxMap(
        float* h_pts,
        unsigned short* h_grid_idxs,
        int pt_num,
        int pt_stride,
        float sample_stride
);

struct GridIdx {
    int x,y,z;
    bool operator==(const GridIdx idx) const
    {
        return this->x==idx.x&&this->y==idx.y&&this->z==idx.z;
    }
};


void hash_combine(std::size_t& seed, std::size_t value) {
    seed ^= value + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

struct GridIdxHasher {
    std::size_t operator()(const GridIdx& idx) const {
        std::size_t seed = 0;
        hash_combine(seed, std::hash<int>()(idx.x));
        hash_combine(seed, std::hash<int>()(idx.y));
        hash_combine(seed, std::hash<int>()(idx.z));

        return seed;
    }
};

void gridDownSample(
        float* pts,
        int pt_num,
        int pt_stride,
        float sample_stride,
        std::vector<int>& downsample_indices
)
{
    //time_t begin=clock();
    unsigned short* grid_idxs=new unsigned short[pt_num*3];
    gridDownSampleIdxMap(pts,grid_idxs,pt_num,pt_stride,sample_stride);
    //std::cout<<"gpu map "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;

    //begin=clock();
    std::unordered_map<unsigned long long, std::vector<int>> map;
    for(int i=0;i<pt_num;i++)
    {
        unsigned long long x=grid_idxs[i*3];
        unsigned long long y=grid_idxs[i*3+1];
        unsigned long long z=grid_idxs[i*3+2];
        unsigned long long idx=0;
        idx=(idx|x)|(y<<16|z<<32);

        auto it=map.find(idx);
        if(it!=map.end())
        {
            it->second.push_back(i);
        }
        else
        {
            std::vector<int> pt_idxs;
            pt_idxs.push_back(i);
            map[idx]=pt_idxs;
        }
    }
    //std::cout<<"sort map "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;

    //begin=clock();
    for(auto it=map.begin();it!=map.end();it++)
    {
        int select_idx=rand()%it->second.size();
        downsample_indices.push_back(it->second[select_idx]);
    }
    //std::cout<<"push back "<<float(clock()-begin)/CLOCKS_PER_SEC<<std::endl;

    delete[] grid_idxs;
}