#include "cuda_common.h"

__global__
void point2VoxelKernel(float* batch_points,float* batch_voxels,float stride,int split_num,int point_num)
{
    int split_num_2=split_num*split_num;

    int pt_index = threadIdx.y + blockIdx.y*blockDim.y;
    if(pt_index>point_num)
        return;

    int batch_index = blockIdx.x;
    float* points=&batch_points[batch_index*point_num*3];
    float* voxels=&batch_voxels[batch_index*split_num*split_num*split_num];


    float x=points[pt_index*3];
    float y=points[pt_index*3+1];
    float z=points[pt_index*3+2];

    int xi=floor(x/stride);
    int yi=floor(y/stride);
    int zi=floor(z/stride);


    float x_ratio=((xi+1)*stride-x)/stride;
    float y_ratio=((yi+1)*stride-y)/stride;
    float z_ratio=((zi+1)*stride-z)/stride;

    if(xi<split_num&&yi<split_num&&zi<split_num)
    {
        int index=xi*split_num_2+yi*split_num+zi;
        float ratio=x_ratio*y_ratio*z_ratio;
        atomicAdd(&voxels[index],ratio);
    }

    if(xi+1<split_num&&yi<split_num&&zi<split_num)
    {
        int index=(xi+1)*split_num_2+yi*split_num+zi;
        float ratio=(1-x_ratio)*y_ratio*z_ratio;
        atomicAdd(&voxels[index],ratio);
    }


    if(xi<split_num&&yi+1<split_num&&z<split_num)
    {
        int index=xi*split_num_2+(yi+1)*split_num+zi;
        float ratio=x_ratio*(1-y_ratio)*z_ratio;
        atomicAdd(&voxels[index],ratio);
    }

    if(xi<split_num&&yi<split_num&&zi+1<split_num)
    {
        int index=xi*split_num_2+yi*split_num+zi+1;
        float ratio=x_ratio*y_ratio*(1-z_ratio);
        atomicAdd(&voxels[index],ratio);
    }

    if(xi+1<split_num&&yi+1<split_num&&zi<split_num)
    {
        int index=(xi+1)*split_num_2+(yi+1)*split_num+zi;
        float ratio=(1-x_ratio)*(1-y_ratio)*z_ratio;
        atomicAdd(&voxels[index],ratio);
    }


    if(xi+1<split_num&&yi<split_num&&zi+1<split_num)
    {
        int index=(xi+1)*split_num_2+yi*split_num+zi+1;
        float ratio=(1-x_ratio)*y_ratio*(1-z_ratio);
        atomicAdd(&voxels[index],ratio);
    }

    if(xi<split_num&&yi+1<split_num&&zi+1<split_num)
    {
        int index=xi*split_num_2+(yi+1)*split_num+zi+1;
        float ratio=x_ratio*(1-y_ratio)*(1-z_ratio);
        atomicAdd(&voxels[index],ratio);
    }

    if(xi+1<split_num&&yi+1<split_num&&zi+1<split_num)
    {
        int index=(xi+1)*split_num_2+(yi+1)*split_num+zi+1;
        float ratio=(1-x_ratio)*(1-y_ratio)*(1-z_ratio);
        atomicAdd(&voxels[index],ratio);
    }
}


__global__
void point2VoxelColorKernel(
        float* batch_points,    // n,k,6
        float* batch_voxels,    // n,s**3,4
        float stride,           // 1.0/(t-1)
        int split_num,          // t
        int point_num           // k
)
{
    int split_num_2=split_num*split_num;

    int pt_index = threadIdx.y + blockIdx.y*blockDim.y;
    int batch_index = blockIdx.x;
    float* points=&batch_points[batch_index*point_num*6];
    float* voxels=&batch_voxels[batch_index*split_num*split_num*split_num*4];

    if(pt_index>point_num)
        return;

    float x=points[pt_index*6];
    float y=points[pt_index*6+1];
    float z=points[pt_index*6+2];
    float r=points[pt_index*6+3];
    float g=points[pt_index*6+4];
    float b=points[pt_index*6+5];

    int xi=floor(x/stride);
    int yi=floor(y/stride);
    int zi=floor(z/stride);


    float x_ratio=((xi+1)*stride-x)/stride;
    float y_ratio=((yi+1)*stride-y)/stride;
    float z_ratio=((zi+1)*stride-z)/stride;

    if(xi<split_num&&yi<split_num&&zi<split_num)
    {
        int index=(xi*split_num_2+yi*split_num+zi)*4;
        float ratio=x_ratio*y_ratio*z_ratio;
        float r_ratio=ratio*r;
        float g_ratio=ratio*g;
        float b_ratio=ratio*b;
        atomicAdd(&voxels[index],ratio);
        atomicAdd(&voxels[index+1],r_ratio);
        atomicAdd(&voxels[index+2],g_ratio);
        atomicAdd(&voxels[index+3],b_ratio);
    }

    if(xi+1<split_num&&yi<split_num&&zi<split_num)
    {
        int index=((xi+1)*split_num_2+yi*split_num+zi)*4;
        float ratio=(1-x_ratio)*y_ratio*z_ratio;
        float r_ratio=ratio*r;
        float g_ratio=ratio*g;
        float b_ratio=ratio*b;
        atomicAdd(&voxels[index],ratio);
        atomicAdd(&voxels[index+1],r_ratio);
        atomicAdd(&voxels[index+2],g_ratio);
        atomicAdd(&voxels[index+3],b_ratio);
    }


    if(xi<split_num&&yi+1<split_num&&z<split_num)
    {
        int index=(xi*split_num_2+(yi+1)*split_num+zi)*4;
        float ratio=x_ratio*(1-y_ratio)*z_ratio;
        float r_ratio=ratio*r;
        float g_ratio=ratio*g;
        float b_ratio=ratio*b;
        atomicAdd(&voxels[index],ratio);
        atomicAdd(&voxels[index+1],r_ratio);
        atomicAdd(&voxels[index+2],g_ratio);
        atomicAdd(&voxels[index+3],b_ratio);
    }

    if(xi<split_num&&yi<split_num&&zi+1<split_num)
    {
        int index=(xi*split_num_2+yi*split_num+zi+1)*4;
        float ratio=x_ratio*y_ratio*(1-z_ratio);
        float r_ratio=ratio*r;
        float g_ratio=ratio*g;
        float b_ratio=ratio*b;
        atomicAdd(&voxels[index],ratio);
        atomicAdd(&voxels[index+1],r_ratio);
        atomicAdd(&voxels[index+2],g_ratio);
        atomicAdd(&voxels[index+3],b_ratio);
    }

    if(xi+1<split_num&&yi+1<split_num&&zi<split_num)
    {
        int index=((xi+1)*split_num_2+(yi+1)*split_num+zi)*4;
        float ratio=(1-x_ratio)*(1-y_ratio)*z_ratio;
        float r_ratio=ratio*r;
        float g_ratio=ratio*g;
        float b_ratio=ratio*b;
        atomicAdd(&voxels[index],ratio);
        atomicAdd(&voxels[index+1],r_ratio);
        atomicAdd(&voxels[index+2],g_ratio);
        atomicAdd(&voxels[index+3],b_ratio);
    }


    if(xi+1<split_num&&yi<split_num&&zi+1<split_num)
    {
        int index=((xi+1)*split_num_2+yi*split_num+zi+1)*4;
        float ratio=(1-x_ratio)*y_ratio*(1-z_ratio);
        float r_ratio=ratio*r;
        float g_ratio=ratio*g;
        float b_ratio=ratio*b;
        atomicAdd(&voxels[index],ratio);
        atomicAdd(&voxels[index+1],r_ratio);
        atomicAdd(&voxels[index+2],g_ratio);
        atomicAdd(&voxels[index+3],b_ratio);
    }

    if(xi<split_num&&yi+1<split_num&&zi+1<split_num)
    {
        int index=(xi*split_num_2+(yi+1)*split_num+zi+1)*4;
        float ratio=x_ratio*(1-y_ratio)*(1-z_ratio);
        float r_ratio=ratio*r;
        float g_ratio=ratio*g;
        float b_ratio=ratio*b;
        atomicAdd(&voxels[index],ratio);
        atomicAdd(&voxels[index+1],r_ratio);
        atomicAdd(&voxels[index+2],g_ratio);
        atomicAdd(&voxels[index+3],b_ratio);
    }

    if(xi+1<split_num&&yi+1<split_num&&zi+1<split_num)
    {
        int index=((xi+1)*split_num_2+(yi+1)*split_num+zi+1)*4;
        float ratio=(1-x_ratio)*(1-y_ratio)*(1-z_ratio);
        float r_ratio=ratio*r;
        float g_ratio=ratio*g;
        float b_ratio=ratio*b;
        atomicAdd(&voxels[index],ratio);
        atomicAdd(&voxels[index+1],r_ratio);
        atomicAdd(&voxels[index+2],g_ratio);
        atomicAdd(&voxels[index+3],b_ratio);
    }
}

void points2VoxelBatch(
        float* batch_points,    // n,k,3
        float* batch_voxels,    // n,s**3
        int split_num,          // s
        int point_num,          // k
        int batch_size,          // n
        int gpu_index

)
{
    HANDLE_ERROR(cudaSetDevice(gpu_index),"set gpu index error")

    int block_num=point_num/1024;
    if(point_num%1024>0) block_num++;
    dim3 block_dim(batch_size,block_num);
    dim3 thread_dim(1,1024);

    float stride=1.f/(split_num-1);
    int voxel_num=split_num*split_num*split_num;

    float* d_batch_points;
    gpuErrchk(cudaMalloc((void**)&d_batch_points, batch_size * point_num * 3 * sizeof(float)))
    gpuErrchk(cudaMemcpy(d_batch_points, batch_points, batch_size * point_num * 3 * sizeof(float),cudaMemcpyHostToDevice))

    float* d_batch_voxels;
    gpuErrchk(cudaMalloc((void**)&d_batch_voxels, batch_size * voxel_num * sizeof(float)))
    gpuErrchk(cudaMemset(d_batch_voxels, 0, batch_size * voxel_num * sizeof(float)))


    point2VoxelKernel<<<block_dim,thread_dim>>>(d_batch_points,d_batch_voxels,stride,split_num,point_num);

    gpuErrchk(cudaMemcpy(batch_voxels, d_batch_voxels, batch_size * voxel_num * sizeof(float),
                            cudaMemcpyDeviceToHost))

    cudaFree(d_batch_points);
    cudaFree(d_batch_voxels);
}


void points2VoxelColorBatch(
        float* batch_points,    // n,k,6
        float* batch_voxels,    // n,s**3
        int split_num,          // s
        int point_num,          // k
        int batch_size,          // n
        int gpu_index

)
{
    HANDLE_ERROR(cudaSetDevice(gpu_index),"set gpu index error")

    int block_num=point_num/1024;
    if(point_num%1024>0) block_num++;
    dim3 block_dim(batch_size,block_num);
    dim3 thread_dim(1,1024);

    float stride=1.f/(split_num-1);
    int voxel_num=split_num*split_num*split_num;

    float* d_batch_points;
    HANDLE_ERROR(cudaMalloc((void**)&d_batch_points, batch_size * point_num * 6 * sizeof(float)),
                 "batch points allocate error")
    HANDLE_ERROR(cudaMemcpy(d_batch_points, batch_points, batch_size * point_num * 6 * sizeof(float),
                            cudaMemcpyHostToDevice),"points copy error")

    float* d_batch_voxels;
    HANDLE_ERROR(cudaMalloc((void**)&d_batch_voxels, batch_size * voxel_num * 4 * sizeof(float)),
                 "batch voxels allocate error")
    HANDLE_ERROR(cudaMemset(d_batch_voxels, 0, batch_size * voxel_num * 4 * sizeof(float)),
                 "batch voxels init error")


    point2VoxelColorKernel<<<block_dim,thread_dim>>>(d_batch_points,d_batch_voxels,stride,split_num,point_num);

    HANDLE_ERROR(cudaMemcpy(batch_voxels, d_batch_voxels, batch_size * voxel_num * 4 * sizeof(float),
                            cudaMemcpyDeviceToHost),"dists copy error")

    cudaFree(d_batch_points);
    cudaFree(d_batch_voxels);
}