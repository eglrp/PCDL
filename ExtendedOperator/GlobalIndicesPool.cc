//
// Created by pal on 17-12-31.
//

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("GlobalIndicesPool")
.Input("global_feature: float32")
.Input("global_indices: int64")
.Output("pooled_feature: float32")
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ::tensorflow::shape_inference::ShapeHandle feature_shape;     // n,f
            ::tensorflow::shape_inference::ShapeHandle indices_shape;     // b,k
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0),2,&feature_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1),2,&indices_shape));

            std::initializer_list<shape_inference::DimensionOrConstant> dims=
                    {c->Dim(indices_shape,0),
                     c->Dim(indices_shape,1),
                     c->Dim(feature_shape,1)};
            c->set_output(0,c->MakeShape(dims));    // b,k,f

            return Status::OK();
        });

class GlobalIndicesPoolOp:public OpKernel
{
public:
    explicit GlobalIndicesPoolOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor& global_feature=context->input(0);    // [n,f]
        const Tensor& global_indices=context->input(1);    // [b,k] max_val=patch_num-1

        OP_REQUIRES(context,global_feature.dims()==2,
                    errors::InvalidArgument("input features must be 2 dimensions"))
        OP_REQUIRES(context,global_indices.dims()==2,
                    errors::InvalidArgument("input indices must be 2 dimensions"))

        int64 n=global_feature.shape().dim_size(0),
              f=global_feature.shape().dim_size(1),
              b=global_indices.shape().dim_size(0),
              k=global_indices.shape().dim_size(1);

        // allocate memory for output
        std::initializer_list<int64> dim_size={b,k,f};
        TensorShape pooled_shape(dim_size);
        Tensor* pooled_feature=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,pooled_shape,&pooled_feature));

        auto feature_tensor=global_feature.shaped<float,2>({n,f});
        auto indices_tensor=global_indices.shaped<int64,2>({b,k});
        auto pooled_tensor=pooled_feature->shaped<float,3>({b,k,f});

        for(int i=0;i<b;i++)
        {
            for(int j=0;j<k;j++)
            {
                // check size
                OP_REQUIRES(context,indices_tensor(i,j)<n,
                            errors::InvalidArgument("Indices must be less than the size of feature's first dimension"));
                memcpy(reinterpret_cast<float*>(&pooled_tensor(i,j,0)),
                       reinterpret_cast<const float*>(&feature_tensor(indices_tensor(i,j),0)),
                       sizeof(float)*f);
            }
        }

    }
};

REGISTER_KERNEL_BUILDER(Name("GlobalIndicesPool").Device(DEVICE_CPU), GlobalIndicesPoolOp);