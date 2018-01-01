//
// Created by pal on 17-12-31.
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("ContextBlockPool")
        .Input("context_feature: float32")
        .Input("context_batch_indices: int64")
        .Input("context_block_indices: int64")
        .Output("pooled_feature: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle feature_shape;           // n,f
            ::tensorflow::shape_inference::ShapeHandle batch_indices_shape;     // b,
            ::tensorflow::shape_inference::ShapeHandle block_indices_shape;     // b,k
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feature_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),1,&batch_indices_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2),2,&block_indices_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(block_indices_shape,0),
                     c->Dim(block_indices_shape,1),
                     c->Dim(feature_shape,1)};
            c->set_output(0,c->MakeShape(dims));    // b,k,f

            return Status::OK();
        });

class ContextBlockPoolOp:public OpKernel
{
public:
    explicit ContextBlockPoolOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor& context_feature=context->input(0);          // [n,f]
        const Tensor& context_batch_indices=context->input(1);    // [b,]
        const Tensor& context_block_indices=context->input(2);    // [b,k]

        OP_REQUIRES(context,context_feature.dims()==2,
                    errors::InvalidArgument("input features must be 2 dimensions"))
        OP_REQUIRES(context,context_batch_indices.dims()==1,
                    errors::InvalidArgument("input batch indices must be 1 dimensions"))
        OP_REQUIRES(context,context_block_indices.dims()==2,
                    errors::InvalidArgument("input block indices must be 2 dimensions"))

        int64 n=context_feature.shape().dim_size(0),
                f=context_feature.shape().dim_size(1),
                b=context_block_indices.shape().dim_size(0),
                k=context_block_indices.shape().dim_size(1);

        OP_REQUIRES(context,context_batch_indices.dim_size(0)==b,
                    errors::InvalidArgument("context_batch_indices are incompatible"));

        // allocate memory for output
        std::initializer_list<int64> dim_size={b,k,f};
        TensorShape pooled_shape(dim_size);
        Tensor* pooled_feature=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,pooled_shape,&pooled_feature));

        auto feature_tensor=context_feature.shaped<float,2>({n,f});
        auto batch_indices_tensor=context_batch_indices.shaped<int64,1>({b});
        auto block_indices_tensor=context_block_indices.shaped<int64,2>({b,k});
        auto pooled_tensor=pooled_feature->shaped<float,3>({b,k,f});

        for(int i=0;i<b;i++)
        {
            for(int j=0;j<k;j++)
            {
                // check size
                auto context_index=batch_indices_tensor(i)+block_indices_tensor(i,j);
                OP_REQUIRES(context,context_index<n,
                            errors::InvalidArgument("index must be less than the size of feature's first dimension"));
                memcpy(reinterpret_cast<float*>(&pooled_tensor(i,j,0)),
                       reinterpret_cast<const float*>(&(feature_tensor(context_index,0))),
                       sizeof(float)*f);
            }
        }

    }
};

REGISTER_KERNEL_BUILDER(Name("ContextBlockPool").Device(DEVICE_CPU), ContextBlockPoolOp);