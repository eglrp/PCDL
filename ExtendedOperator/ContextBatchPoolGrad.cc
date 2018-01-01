//
// Created by pal on 17-12-31.
//


#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("ContextBatchPoolGrad")
        .Input("context_feature: float32")
        .Input("context_batch_indices: int64")
        .Input("pooled_feature_grad: float32")
        .Output("context_feature_grad: float32");

class ContextBatchPoolGradOp:public OpKernel
{
public:
    explicit ContextBatchPoolGradOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor& context_feature=context->input(0);          // [n,f]
        const Tensor& context_batch_indices=context->input(1);    // [b,]
        const Tensor& pooled_feature_grad=context->input(2);      // [b,f]

        OP_REQUIRES(context,context_feature.dims()==2,
                    errors::InvalidArgument("input features must be 2 dimensions"))
        OP_REQUIRES(context,context_batch_indices.dims()==1,
                    errors::InvalidArgument("input batch indices must be 1 dimensions"))
        OP_REQUIRES(context,pooled_feature_grad.dims()==2,
                    errors::InvalidArgument("input pooled feature grad must be 2 dimensions"))

        int64 n=context_feature.shape().dim_size(0),
                f=context_feature.shape().dim_size(1),
                b=context_batch_indices.shape().dim_size(0);

        OP_REQUIRES(context,pooled_feature_grad.dim_size(0)==b&&
                            pooled_feature_grad.dim_size(1)==f,
                    errors::InvalidArgument("context_batch_indices are incompatible"));

        // allocate memory for output
        std::initializer_list<int64> dim_size={n,f};
        TensorShape feature_shape(dim_size);
        Tensor* feature_grad=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,feature_shape,&feature_grad));

        auto context_feature_tensor=context_feature.shaped<float,2>({n,f});
        auto batch_indices_tensor=context_batch_indices.shaped<int64,1>({b});
        auto pooled_grad_tensor=pooled_feature_grad.shaped<float,2>({b,f});
        auto feature_grad_tensor=feature_grad->shaped<float,2>({n,f});

        // initialize to 0
        for(int i=0;i<n;i++)
            for(int j=0;j<f;j++)
                feature_grad_tensor(i,j)=0;

        //
        for(int i=0;i<b;i++)
        {
            auto begin=batch_indices_tensor(i);
            long end;
            if(i==b-1) end=n;
            else  end=batch_indices_tensor(i+1);
            std::vector<float> max_val(static_cast<unsigned int>(f),std::numeric_limits<float>::lowest());
            std::vector<int> max_index(static_cast<unsigned int>(f));
            for(auto j=begin;j<end;j++)
                for(int l=0;l<f;l++)
                {
                    if(max_val[l]<context_feature_tensor(j,l))
                    {
                        max_val[l]=context_feature_tensor(j,l);
                        max_index[l]= static_cast<int>(j);
                    }
                }

            for(int l=0;l<f;l++)
                feature_grad_tensor(max_index[l],l)+=pooled_grad_tensor(i,l);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ContextBatchPoolGrad").Device(DEVICE_CPU), ContextBatchPoolGradOp);