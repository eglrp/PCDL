//
// Created by pal on 17-12-31.
//


#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("GlobalIndicesPoolGrad")
        .Input("global_feature: float32")
        .Input("global_indices: int64")
        .Input("pooled_grad: float32")
        .Output("global_feature_grad: float32");

class GlobalIndicesPoolGradOp:public OpKernel
{
public:
    explicit GlobalIndicesPoolGradOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor& global_feature=context->input(0);    // [n,f]
        const Tensor& global_indices=context->input(1);    // [b,k]
        const Tensor& pooled_grad=context->input(2);    //  [b,k,f]

        OP_REQUIRES(context,global_feature.dims()==2,
                    errors::InvalidArgument("input features must be 2 dimensions"));
        OP_REQUIRES(context,global_indices.dims()==2,
                    errors::InvalidArgument("input indices must be 2 dimensions"));
        OP_REQUIRES(context,pooled_grad.dims()==3,
                    errors::InvalidArgument("pooled grad must be 3 dimensions"));

        int64 n=global_feature.shape().dim_size(0),
                f=global_feature.shape().dim_size(1),
                b=global_indices.shape().dim_size(0),
                k=global_indices.shape().dim_size(1);

        OP_REQUIRES(context,pooled_grad.dim_size(0)==b&&
                            pooled_grad.dim_size(1)==k&&
                            pooled_grad.dim_size(2)==f,
                            errors::InvalidArgument("pooled grad is incompatible"));

        // allocate memory for output
        std::initializer_list<int64> dim_size={n,f};
        TensorShape feature_shape(dim_size);
        Tensor* feature_grad=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,feature_shape,&feature_grad));

        auto indices_tensor=global_indices.shaped<int64,2>({b,k});
        auto pooled_grad_tensor=pooled_grad.shaped<float,3>({b,k,f});
        auto feature_grad_tensor=feature_grad->shaped<float,2>({n,f});

        // initialize to 0
        for(int i=0;i<n;i++)
            for(int j=0;j<f;j++)
                feature_grad_tensor(i,j)=0;

        // assign value
        for(int i=0;i<b;i++)
            for(int j=0;j<k;j++)
                for(int l=0;l<f;l++)
                    feature_grad_tensor(indices_tensor(i,j),l)+=pooled_grad_tensor(i,j,l);
    }
};

REGISTER_KERNEL_BUILDER(Name("GlobalIndicesPoolGrad").Device(DEVICE_CPU), GlobalIndicesPoolGradOp);