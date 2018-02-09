#include <vector>
#include <cstdlib>
#include <cmath>
#include <iostream>

std::vector<std::vector<int> >
randomSample(
        float* points,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float block_size,
        float maxx,
        float maxy,
        float min_ratio,
        int split_num,
        float& rot_angle_out
)
{
    float rot_angle=float(rand()%90)/180*M_PI;
    rot_angle_out=-rot_angle;
    float block_axis_xx=cos(rot_angle);
    float block_axis_xy=sin(rot_angle);
    float block_axis_yx=-block_axis_xy;
    float block_axis_yy=block_axis_xx;

    //std::cout<<"begin\n";
    std::vector<float> sample_origin;
    float x=-block_size;
    while(x<=maxx)
    {
        x+=sample_stride;
        float y=-block_size;
        while(y<=maxy)
        {
            y+=sample_stride;
            sample_origin.push_back(x);
            sample_origin.push_back(y);
        }
    }

    // sample many points in each block,
    // if too few points inside the original point region,
    // then discard this block
    int origin_num=sample_origin.size()/2;
    std::vector<float> retain_origin;
    float test_interval=block_size/split_num;
    //time_t begin=clock();
    for(int i=0;i<origin_num;i++)
    {
        float ox=sample_origin[i*2];
        float oy=sample_origin[i*2+1];
        int inside_num=0;
        for(int j=0;j<split_num;j++)
            for(int k=0;k<split_num;k++)
            {
                float test_x=ox+j*test_interval*block_axis_xx+k*test_interval*block_axis_yx;
                float test_y=oy+j*test_interval*block_axis_xy+k*test_interval*block_axis_yy;
                if(test_x<=maxx&&test_x>=0&&
                   test_y<=maxy&&test_y>=0)
                    inside_num++;
            }

        if(inside_num/float(split_num*split_num)>=min_ratio)
        {
            retain_origin.push_back(ox);
            retain_origin.push_back(oy);
        }
    }
    //std::cout<<"validate cost "<<float(clock()-begin)/CLOCKS_PER_SEC<<"s \n";

    //std::cout<<"validate\n";
    int retain_origin_num=retain_origin.size()/2;
    std::vector<std::vector<int> > block_idxs(retain_origin_num);

    //begin=clock();
    for(int i=0;i<pt_num;i++)
    {
        float x=points[i*pt_stride+0];
        float y=points[i*pt_stride+1];
        for(int j=0;j<retain_origin_num;j++)
        {
            float ox=retain_origin[j*2];
            float oy=retain_origin[j*2+1];
            float diff_x=x-ox;
            float diff_y=y-oy;
            float proj_x=diff_x*block_axis_xx+diff_y*block_axis_xy;
            float proj_y=diff_x*block_axis_yx+diff_y*block_axis_yy;
            if(proj_x>=0&&proj_x<=block_size&&proj_y>=0&&proj_y<=block_size)
                block_idxs[j].push_back(i);
        }
    }
    //std::cout<<"gather cost "<<float(clock()-begin)/CLOCKS_PER_SEC<<"s \n";
    //std::cout<<"gather\n";

    return block_idxs;
}

void randomSampleGatherPointsGPU(
        float* points,         // [pt_num,pt_stride] xyz rgb ...
        float* retain_origin,  // [retain_num,2] xy
        bool* result,          // [pt_num,retain_num] 0 or 1
        int pt_stride,
        int pt_num,
        int retain_num,
        float block_axis_xx,
        float block_axis_xy,
        float block_axis_yx,
        float block_axis_yy,
        float block_size,
        int gpu_index
);


std::vector<std::vector<int> >
randomSampleGPU(
        float* points,
        int pt_num,
        int pt_stride,
        float sample_stride,
        float block_size,
        float maxx,
        float maxy,
        float min_ratio,
        int split_num,
        float& rot_angle_out,
        int gpu_index
)
{
    float rot_angle=float(rand()%90)/180*M_PI;
    rot_angle_out=-rot_angle;
    float block_axis_xx=cos(rot_angle);
    float block_axis_xy=sin(rot_angle);
    float block_axis_yx=-block_axis_xy;
    float block_axis_yy=block_axis_xx;

    //std::cout<<"begin\n";
    std::vector<float> sample_origin;
    float x=-block_size;
    while(x<=maxx)
    {
        x+=sample_stride;
        float y=-block_size;
        while(y<=maxy)
        {
            y+=sample_stride;
            sample_origin.push_back(x);
            sample_origin.push_back(y);
        }
    }

    // sample many points in each block,
    // if too few points inside the original point region,
    // then discard this block
    int origin_num=sample_origin.size()/2;
    std::vector<float> retain_origin;
    float test_interval=block_size/split_num;
    //time_t begin=clock();
    for(int i=0;i<origin_num;i++)
    {
        float ox=sample_origin[i*2];
        float oy=sample_origin[i*2+1];
        int inside_num=0;
        for(int j=0;j<split_num;j++)
            for(int k=0;k<split_num;k++)
            {
                float test_x=ox+j*test_interval*block_axis_xx+k*test_interval*block_axis_yx;
                float test_y=oy+j*test_interval*block_axis_xy+k*test_interval*block_axis_yy;
                if(test_x<=maxx&&test_x>=0&&
                   test_y<=maxy&&test_y>=0)
                    inside_num++;
            }

        if(inside_num/float(split_num*split_num)>=min_ratio)
        {
            retain_origin.push_back(ox);
            retain_origin.push_back(oy);
        }
    }
    //std::cout<<"validate cost "<<float(clock()-begin)/CLOCKS_PER_SEC<<"s \n";

    //std::cout<<"validate\n";
    int retain_origin_num=retain_origin.size()/2;

    bool* result=new bool[pt_num*retain_origin_num];
    randomSampleGatherPointsGPU(points,retain_origin.data(),result,
                                pt_stride,pt_num,retain_origin_num,
                                block_axis_xx,block_axis_xy,
                                block_axis_yx,block_axis_yy,block_size,gpu_index);


    std::vector<std::vector<int> > block_idxs(retain_origin_num);

    //begin=clock();
    for(int i=0;i<pt_num;i++)
    {
        for(int j=0;j<retain_origin_num;j++)
        {
            if(result[i*retain_origin_num+j])
                block_idxs[j].push_back(i);
        }
    }

    //std::cout<<"gather cost "<<float(clock()-begin)/CLOCKS_PER_SEC<<"s \n";
    //std::cout<<"gather\n";

    delete [] result;

    return block_idxs;
}