#include "cuda_common.h"

__global__
void computeCovarsBatchKernel(float* batch_points,int* batch_nidxs,float* batch_covars,int nn_size,int point_num)
{
    int pt_index = threadIdx.y + blockIdx.y*blockDim.y;
    if(pt_index>point_num)
        return;

    int batch_index = blockIdx.x;
    float* points=&batch_points[batch_index*point_num*3];
    int* idxs=&batch_nidxs[batch_index*point_num*nn_size];
    float* covars=&batch_covars[batch_index*point_num*9];

    float sx=0.f,sy=0.f,sz=0.f;
    float sxy=0.f,sxz=0.f,syz=0.f;
    float sxx=0.f,syy=0.f,szz=0.f;
    for(int i=0;i<nn_size;i++)
    {
        int cur_pt_index=idxs[pt_index*nn_size+i];
        float x=points[cur_pt_index*3];
        float y=points[cur_pt_index*3+1];
        float z=points[cur_pt_index*3+2];
        sx+=x;sy+=y;sz+=z;
        sxy+=x*y;sxz+=x*z;syz+=y*z;
        sxx+=x*x;syy+=y*y;szz+=z*z;
    }
    float x=points[pt_index*3];
    float y=points[pt_index*3+1];
    float z=points[pt_index*3+2];
    sx+=x;sy+=y;sz+=z;
    sxy+=x*y;sxz+=x*z;syz+=y*z;
    sxx+=x*x;syy+=y*y;szz+=z*z;

    int n=(nn_size+1);
    float cxx=sxx/n-sx/n*sx/n;
    float cyy=syy/n-sy/n*sy/n;
    float czz=szz/n-sz/n*sz/n;
    float cxy=sxy/n-sx/n*sy/n;
    float cxz=sxz/n-sx/n*sz/n;
    float cyz=syz/n-sy/n*sz/n;

    //normalize
    float norm=sqrt(cxx*cxx+cyy*cyy+czz*czz+cxy*cxy*2+cxz*cxz*2+cyz*cyz*2);
    if(norm>1e-6)
    {
        cxx/=norm;
        cyy/=norm;
        czz/=norm;
        cxy/=norm;
        cxz/=norm;
        cyz/=norm;
    }

    covars[pt_index*9]=cxx;
    covars[pt_index*9+1]=cxy;
    covars[pt_index*9+2]=cxz;
    covars[pt_index*9+3]=cxy;
    covars[pt_index*9+4]=cyy;
    covars[pt_index*9+5]=cyz;
    covars[pt_index*9+6]=cxz;
    covars[pt_index*9+7]=cyz;
    covars[pt_index*9+8]=czz;
}



void computeCovarsBatch(
        float* batch_points,    // n,k,3
        int* batch_idxs,        // n,k,t
        float* batch_covars,    // n,k,9
        int nn_size,            // t
        int point_num,          // k
        int batch_size,         // n
        int gpu_index
)
{
    HANDLE_ERROR(cudaSetDevice(gpu_index),"set gpu index error")

    int block_num=point_num/1024;
    if(point_num%1024>0) block_num++;
    dim3 block_dim(batch_size,block_num);
    dim3 thread_dim(1,1024);

    float* d_batch_points;
    HANDLE_ERROR(cudaMalloc((void**)&d_batch_points, batch_size * point_num * 3 * sizeof(float)),
                 "batch points allocate error")
    HANDLE_ERROR(cudaMemcpy(d_batch_points, batch_points, batch_size * point_num * 3 * sizeof(float),
                            cudaMemcpyHostToDevice),"points copy error")

    int* d_batch_idxs;
    HANDLE_ERROR(cudaMalloc((void**)&d_batch_idxs, batch_size * point_num * nn_size * sizeof(int)),
                 "batch points allocate error")
    HANDLE_ERROR(cudaMemcpy(d_batch_idxs, batch_idxs, batch_size * point_num * nn_size * sizeof(int),
                            cudaMemcpyHostToDevice),"nidxs copy error")

    float* d_batch_covars;
    HANDLE_ERROR(cudaMalloc((void**)&d_batch_covars, batch_size * point_num * 9 * sizeof(float)),
                 "batch covars allocate error")


    computeCovarsBatchKernel<<<block_dim,thread_dim>>>(d_batch_points,d_batch_idxs,d_batch_covars,nn_size,point_num);

    gpuErrchk(cudaMemcpy(batch_covars, d_batch_covars, batch_size * point_num * 9 * sizeof(float),
                            cudaMemcpyDeviceToHost));

    cudaFree(d_batch_points);
    cudaFree(d_batch_idxs);
    cudaFree(d_batch_covars);
}
