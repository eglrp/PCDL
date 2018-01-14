#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cfloat>
#include <cstdio>

#define HANDLE_ERROR(func,message) if((func)!=cudaSuccess) { printf("%s \n",message);   return; }

__global__
void findNearest(float* points,float* dists,unsigned int* idxs,int point_num)
{
    int c_idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(c_idx>=point_num)
        return;

    float c_x=points[c_idx];
    float c_y=points[c_idx+1];
    float c_z=points[c_idx+2];

    float min_dist=FLT_MAX;
    unsigned int min_idx=0;
    for(unsigned int s_idx=0;s_idx<point_num;s_idx++)
    {
        if(s_idx==c_idx) continue;

        float s_x=points[s_idx];
        float s_y=points[s_idx+1];
        float s_z=points[s_idx+2];
        float dist=(c_x-s_x)*(c_x-s_x)+(c_y-s_y)*(c_y-s_y)+(c_z-s_z)*(c_z-s_z);
        if(dist<min_dist)
        {
            min_dist=dist;
            min_idx=s_idx;
        }
    }
    dists[c_idx]=sqrt(min_dist);
    idxs[c_idx]=min_idx;
}

void findNearestPoints(
    float* points,
    unsigned int point_num,
    float* dists,
    unsigned int* idxs
)
{
    int block_num=point_num/1024;
    if(point_num%1024>0) block_num++;

    float* d_points;
    HANDLE_ERROR(cudaMalloc((void**)&d_points, point_num * 3 * sizeof(float)),"allocate error")
    HANDLE_ERROR(cudaMemcpy(d_points, points, point_num * 3 * sizeof(float), cudaMemcpyHostToDevice),"points copy error")

    float* d_dists;
    HANDLE_ERROR(cudaMalloc((void**)&d_dists, point_num * sizeof(float)),"allocate error")

    unsigned int* d_idxs;
    HANDLE_ERROR(cudaMalloc((void**)&d_idxs, point_num * sizeof(unsigned int)),"allocate error")

    findNearest<<<block_num,1024>>>(d_points,d_dists,d_idxs,point_num);

    HANDLE_ERROR(cudaMemcpy(dists, d_dists, point_num * sizeof(float), cudaMemcpyDeviceToHost),"dists copy error")
    HANDLE_ERROR(cudaMemcpy(idxs, d_idxs, point_num * sizeof(unsigned int), cudaMemcpyDeviceToHost),"idxs copy error")
}