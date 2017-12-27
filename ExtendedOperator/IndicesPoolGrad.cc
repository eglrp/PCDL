//
// Created by pal on 17-12-25.
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <iostream>

using namespace tensorflow;

REGISTER_OP("IndicesPoolGrad")
        .Attr("patch_num: int")
        .Input("feature: float32")
        .Input("indices: int64")
        .Input("pool_feature_grad: float32")
        .Output("feature_grad: float32");



class IndicesPoolGradOp:public OpKernel {
    int32 patch_num;
public:
    explicit IndicesPoolGradOp(OpKernelConstruction *context) : OpKernel(context) {
        context->GetAttr("patch_num", &patch_num);
    }

    void Compute(OpKernelContext *context) override {
        // fetch input tensor
        const Tensor& feature=context->input(0);                // [n,k,f]
        const Tensor& indices=context->input(1);                // [n,k] max_val=patch_num-1
        const Tensor& pooled_feature_grad=context->input(2);    // [n,patch_num,f] max_val=patch_num-1
        int64 n=feature.shape().dim_size(0),
                k=feature.shape().dim_size(1),
                f=feature.shape().dim_size(2);

        // pooling operator
        auto feature_tensor=feature.shaped<float,3>({n,k,f});
        auto indices_tensor=indices.shaped<int64,2>({n,k});
        auto pooled_grad_tensor=pooled_feature_grad.shaped<float,3>({n,patch_num,f});

        Tensor* feature_grad=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,feature.shape(),&feature_grad));

        auto feature_grad_tensor=feature_grad->shaped<float,3>({n,k,f});


        // count the number
        std::vector<std::vector<int>> counts(static_cast<unsigned int>(n),
                                             std::vector<int>(static_cast<unsigned int>(patch_num),0));
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<k;j++)
            {
                counts[i][indices_tensor(i,j)]++;
            }
        }

        // initialize to 0
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<k;j++)
            {
                for(int l=0;l<f;l++)
                {
                    feature_grad_tensor(i,j,l)=0.0f;
                }
            }
        }

        for(int i=0;i<n;i++)
        {
            std::vector<std::vector<int>> assign_indices(static_cast<unsigned long>(patch_num),
                                             std::vector<int>(static_cast<unsigned long>(f)));
            std::vector<std::vector<float>> max_val(static_cast<unsigned long>(patch_num),
                    std::vector<float>(static_cast<unsigned long>(f),std::numeric_limits<float>::lowest()));
            // log the index
            for(int j=0;j<k;j++)
            {
                for(int l=0;l<f;l++)
                {
                    float* loc=&max_val[indices_tensor(i,j)][l];
                    if(*loc<feature_tensor(i,j,l))
                    {
                        *loc=feature_tensor(i,j,l);
                        assign_indices[indices_tensor(i,j)][l]=j;
                    }
                }
            }

            // write grad
            for(int j=0;j<patch_num;j++)
            {
                if(counts[i][j]==0)
                    continue;
                for(int l=0;l<f;l++)
                {
                    feature_grad_tensor(i,assign_indices[j][l],l)=pooled_grad_tensor(i,j,l);
                }
            }
        }

    }
};

REGISTER_KERNEL_BUILDER(Name("IndicesPoolGrad").Device(DEVICE_CPU), IndicesPoolGradOp);