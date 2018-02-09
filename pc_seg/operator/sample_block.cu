#include "cuda_common.h"

__global__
void gatherPoints(float* points,
                  float* retain_origin,
                  bool* result,
                  int point_stride,
                  int point_num,
                  int retain_num,
                  float block_axis_xx,
                  float block_axis_xy,
                  float block_axis_yx,
                  float block_axis_yy,
                  float block_size)
{
    int pt_index = threadIdx.y + blockIdx.y*blockDim.y;
    if(pt_index>point_num)
        return;

    float x=points[pt_index*point_stride+0];
    float y=points[pt_index*point_stride+1];
    for(int j=0;j<retain_num;j++)
    {
        float ox=retain_origin[j*2];
        float oy=retain_origin[j*2+1];
        float diff_x=x-ox;
        float diff_y=y-oy;
        float proj_x=diff_x*block_axis_xx+diff_y*block_axis_xy;
        float proj_y=diff_x*block_axis_yx+diff_y*block_axis_yy;

        if(proj_x>=0&&proj_x<=block_size&&proj_y>=0&&proj_y<=block_size)
            result[pt_index*retain_num+j]=true;
        else
            result[pt_index*retain_num+j]=false;
    }
}

void randomSampleGatherPointsGPU(
        float* points,         // [point_num,pt_stride] xyz rgb ...
        float* retain_origin,  // [retain_num,2] xy
        bool* result,          // [point_num,retain_num] 0 or 1
        int point_stride,
        int point_num,
        int retain_num,
        float block_axis_xx,
        float block_axis_xy,
        float block_axis_yx,
        float block_axis_yy,
        float block_size,
        int gpu_index
)
{

    HANDLE_ERROR(cudaSetDevice(gpu_index),"set gpu index error")

    int block_num=point_num/1024;
    if(point_num%1024>0) block_num++;
    dim3 block_dim(1,block_num);
    dim3 thread_dim(1,1024);

    float * d_points,*d_retain_origin;
    bool* d_result;
    gpuErrchk(cudaMalloc((void**)&d_points, point_num * point_stride * sizeof(float)))
    gpuErrchk(cudaMalloc((void**)&d_retain_origin, retain_num * 2 * sizeof(float)))
    gpuErrchk(cudaMalloc((void**)&d_result, point_num * retain_num * sizeof(bool)))

    gpuErrchk(cudaMemcpy(d_points, points, point_num * point_stride * sizeof(float), cudaMemcpyHostToDevice))
    gpuErrchk(cudaMemcpy(d_retain_origin, retain_origin, retain_num * 2 * sizeof(float), cudaMemcpyHostToDevice))

    gatherPoints<<<block_dim,thread_dim>>>(
                  d_points,d_retain_origin,d_result,
                  point_stride,point_num,retain_num,
                  block_axis_xx,block_axis_xy,
                  block_axis_yx,block_axis_yy,block_size);

    gpuErrchk(cudaMemcpy(result, d_result, point_num * retain_num * sizeof(bool), cudaMemcpyDeviceToHost))

    cudaFree(d_points);
    cudaFree(d_retain_origin);
    cudaFree(d_result);

}

