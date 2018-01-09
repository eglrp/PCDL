import numpy as np
import matplotlib.pyplot as plt

def output_activation(feature, filename, dim, pts):
    pt_num=pts.shape[0]
    # indices = np.argsort(-feature[:, dim])
    max_feature_val = np.max(feature[:, dim])
    min_feature_val = np.min(feature[:, dim])

    color_count = np.zeros(256,np.float32)
    for i in xrange(pt_num):
        this_color = (feature[i, dim]-min_feature_val) / (max_feature_val-min_feature_val) *255
        color_count[int(this_color)]+=1

    for i in range(1,256):
        color_count[i]+=color_count[i-1]

    color_count/=color_count[-1]

    color = np.random.uniform(0, 1, [3])
    color = color/np.sqrt(np.sum(color**2))
    with open(filename, 'w') as f:
        for i in xrange(pt_num):
                this_color = color_count[int((feature[i, dim]-min_feature_val) /
                                             (max_feature_val-min_feature_val)*255)]*color
                this_color = np.asarray(this_color*255, np.int)
                f.write('{} {} {} {} {} {}\n'.format(
                    pts[i, 0], pts[i, 1], pts[i, 2],
                    this_color[0], this_color[1], this_color[2]))

def output_activation_distribution(feature,dim,filename):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(feature[:,dim].flatten(),bins=100)
    fig.savefig(filename)
    plt.close(fig)

def output_points(filename,pts,colors=None):
    has_color=pts.shape[1]>=6
    with open(filename, 'w') as f:
        for i,pt in enumerate(pts):
            if colors is None:
                if has_color:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],int(pt[3]),int(pt[4]),int(pt[5])))
                else:
                    f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            else:
                if colors.shape[0]==pts.shape[0]:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[i,0],colors[i,1],colors[i,2]))
                else:
                    f.write('{} {} {} {} {} {}\n'.format(pt[0],pt[1],pt[2],colors[0],colors[1],colors[2]))

