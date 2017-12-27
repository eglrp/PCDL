#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/core/framework/op_kernel.h>

using namespace tensorflow;

REGISTER_OP("IndicesPool")
    .Attr("patch_num: int")
    .Input("feature: float32")
    .Input("indices: int64")
    .Output("pooled_feature: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle feature_shape;
        ::tensorflow::shape_inference::ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0),3,&feature_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1),2,&indices_shape));

        int32 patch_num;
        c->GetAttr("patch_num",&patch_num);

        std::initializer_list<shape_inference::DimensionOrConstant> dims=
                {c->Dim(indices_shape,0),patch_num,c->Dim(feature_shape,2)};
        c->set_output(0,c->MakeShape(dims));

        return Status::OK();
    });


class IndicesPoolOp:public OpKernel
{
    int32 patch_num;
public:
    explicit IndicesPoolOp(OpKernelConstruction* context) : OpKernel(context)
    {
        context->GetAttr("patch_num",&patch_num);
    }

    void Compute(OpKernelContext* context) override
    {
        // fetch input tensor
        const Tensor& feature=context->input(0);    // [n,k,f]
        const Tensor& indices=context->input(1);    // [n,k] max_val=patch_num-1
        int64 n=feature.shape().dim_size(0),
                k=feature.shape().dim_size(1),
                f=feature.shape().dim_size(2);

        OP_REQUIRES(context,indices.dim_size(0)==n,
                    errors::InvalidArgument("Indices shape are incompatible with feature shape"));
        OP_REQUIRES(context,indices.dim_size(1)==k,
                    errors::InvalidArgument("Indices shape are incompatible with feature shape"));

        // allocate memory for output
        std::initializer_list<int64> dim_size={n,patch_num,f};

        TensorShape pooled_shape(dim_size);

        Tensor* pooled_feature=NULL;
        OP_REQUIRES_OK(context,context->allocate_output(0,pooled_shape,&pooled_feature));

        // pooling operator
        auto feature_tensor=feature.shaped<float,3>({n,k,f});
        auto indices_tensor=indices.shaped<int64,2>({n,k});
        auto pooled_tensor=pooled_feature->shaped<float,3>({n,patch_num,f});

        // check the result
        std::vector<std::vector<int>> counts(static_cast<unsigned int>(n),
                                             std::vector<int>(static_cast<unsigned int>(patch_num),0));
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<k;j++)
            {
                OP_REQUIRES(context,indices_tensor(i,j)<patch_num,
                            errors::InvalidArgument("Indices must be less than patch_num"));
                counts[i][indices_tensor(i,j)]++;
            }
        }

        //initialization
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<patch_num;j++)
            {
                for(int l=0;l<f;l++)
                {
                    pooled_tensor(i,j,l)=std::numeric_limits<float>::lowest();
                    if(counts[i][j]==0)
                        pooled_tensor(i,j,l)=0.f;
                }
            }
        }

        // pooling
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<k;j++)
            {
                for(int l=0;l<f;l++)
                {
                    float* loc=&pooled_tensor(i,indices_tensor(i,j),l);
                    (*loc)=std::max((*loc),feature_tensor(i,j,l));
                }
            }
        }
    }

};

REGISTER_KERNEL_BUILDER(Name("IndicesPool").Device(DEVICE_CPU), IndicesPoolOp);

