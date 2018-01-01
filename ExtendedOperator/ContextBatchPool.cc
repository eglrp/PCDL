//
// Created by pal on 17-12-31.
//

//
// Created by pal on 17-12-31.
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("ContextBatchPool")
        .Input("context_feature: float32")
        .Input("context_batch_indices: int64")
        .Output("pooled_feature: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle feature_shape;   // n,f
            ::tensorflow::shape_inference::ShapeHandle indices_shape;   // b,
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feature_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&indices_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(indices_shape,0),c->Dim(feature_shape,1)};
            c->set_output(0,c->MakeShape(dims));    // b,f

            return Status::OK();
        });

class ContextBatchPoolOp:public OpKernel
{
public:
    explicit ContextBatchPoolOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor& context_feature=context->input(0);          // [n,f]
        const Tensor& context_batch_indices=context->input(1);    // [b,]

        OP_REQUIRES(context,context_feature.dims()==2,
                    errors::InvalidArgument("input features must be 2 dimensions"))
        OP_REQUIRES(context,context_batch_indices.dims()==1,
                    errors::InvalidArgument("input batch indices must be 1 dimensions"))

        int64 n=context_feature.shape().dim_size(0),
                f=context_feature.shape().dim_size(1),
                b=context_batch_indices.shape().dim_size(0);

        // allocate memory for output
        std::initializer_list<int64> dim_size={b,f};
        TensorShape pooled_shape(dim_size);
        Tensor* pooled_feature=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,pooled_shape,&pooled_feature));

        auto feature_tensor=context_feature.shaped<float,2>({n,f});
        auto batch_indices_tensor=context_batch_indices.shaped<int64,1>({b});
        auto pooled_tensor=pooled_feature->shaped<float,2>({b,f});

        // initialization
        for(int i=0;i<b;i++)
            for(int j=0;j<f;j++)
                pooled_tensor(i,j)=std::numeric_limits<float>::lowest();

        // pool
        for(int i=0;i<b;i++)
        {
            auto begin=batch_indices_tensor(i);
            long end;
            if(i==b-1) end=n;
            else  end=batch_indices_tensor(i+1);
            for(auto j=begin;j<end;j++)
                for(int l=0;l<f;l++)
                    pooled_tensor(i,l)=std::max(pooled_tensor(i,l),feature_tensor(j,l));
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("ContextBatchPool").Device(DEVICE_CPU), ContextBatchPoolOp);