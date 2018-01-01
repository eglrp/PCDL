//
// Created by pal on 17-12-31.
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("ContextBlockPoolGrad")
        .Input("context_feature: float32")
        .Input("context_batch_indices: int64")
        .Input("context_block_indices: int64")
        .Input("pooled_feature_grad: float32")
        .Output("context_feature_grad: float32");

class ContextBlockPoolGradOp:public OpKernel
{
public:
    explicit ContextBlockPoolGradOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor& context_feature=context->input(0);          // [n,f]
        const Tensor& context_batch_indices=context->input(1);    // [b,]
        const Tensor& context_block_indices=context->input(2);    // [b,k]
        const Tensor& pooled_feature_grad=context->input(3);      // [b,k,f]

        OP_REQUIRES(context,context_feature.dims()==2,
                    errors::InvalidArgument("input features must be 2 dimensions"))
        OP_REQUIRES(context,context_batch_indices.dims()==1,
                    errors::InvalidArgument("input batch indices must be 1 dimensions"))
        OP_REQUIRES(context,context_block_indices.dims()==2,
                    errors::InvalidArgument("input block indices must be 2 dimensions"))
        OP_REQUIRES(context,pooled_feature_grad.dims()==3,
                    errors::InvalidArgument("input pooled feature grad must be 3 dimensions"))

        int64 n=context_feature.shape().dim_size(0),
                f=context_feature.shape().dim_size(1),
                b=context_block_indices.shape().dim_size(0),
                k=context_block_indices.shape().dim_size(1);

        OP_REQUIRES(context,context_batch_indices.dim_size(0)==b,
                    errors::InvalidArgument("context_batch_indices are incompatible"));
        OP_REQUIRES(context,pooled_feature_grad.dim_size(0)==b&&
                            pooled_feature_grad.dim_size(1)==k&&
                            pooled_feature_grad.dim_size(2)==f,
                    errors::InvalidArgument("pooled_feature_grad are incompatible"));

        // allocate memory for output
        std::initializer_list<int64> dim_size={n,f};
        TensorShape feature_shape(dim_size);
        Tensor* feature_grad=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,feature_shape,&feature_grad));

        auto batch_indices_tensor=context_batch_indices.shaped<int64,1>({b});
        auto block_indices_tensor=context_block_indices.shaped<int64,2>({b,k});
        auto pooled_grad_tensor=pooled_feature_grad.shaped<float,3>({b,k,f});
        auto feature_grad_tensor=feature_grad->shaped<float,2>({n,f});

        // initialize to 0
        for(int i=0;i<n;i++)
            for(int j=0;j<f;j++)
                feature_grad_tensor(i,j)=0;

        // assign value
        for(int i=0;i<b;i++)
            for(int j=0;j<k;j++)
            {
                auto context_index=batch_indices_tensor(i)+block_indices_tensor(i,j);
                for(int l=0;l<f;l++)
                {
                    feature_grad_tensor(context_index,l)+=pooled_grad_tensor(i,j,l);
                }

            }

    }
};

REGISTER_KERNEL_BUILDER(Name("ContextBlockPoolGrad").Device(DEVICE_CPU), ContextBlockPoolGradOp);