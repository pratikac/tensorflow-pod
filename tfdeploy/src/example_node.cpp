#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "tfutils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        printf("Usage: example_node [img_name] [protobuf]\n");
        exit(1);
    }
    
    char* img_name = argv[1];
    char* pb_name = argv[2];
    auto m = imread(img_name, CV_LOAD_IMAGE_COLOR);
    if(m.empty())
    {
        printf("Could not find image: %s\n", img_name);
        exit(1);
    }
    
    cvtColor(m, m, CV_BGRA2RGB);
    resize(m, m, Size(299,299));

    int64_t xsz[] = {1, 299, 299, 3};
    int64_t ysz[] = {1, 1000, 1, 1};
    
    uint8_t xmean[] = {128, 128, 128};
    uint8_t xstd[] = {128, 128, 128};

    tfdeploy_t tf;
    tf.init_network_opts(pb_name,
            (char*)"Mul", xsz, xmean, xstd,
            (char*)"softmax", ysz);

    for(int i=0; i<100; i++)
    {
        vector<Mat> x = {m};
        vector<float> y;

        tf.run_network(x, y);
        for(auto& yi : y)
            printf("%.3f ", yi);
        printf("\n");
        
        printf("class idx: [%d]\n", distance(y.begin(), max_element(y.begin(), y.end())));
    }
    return 0;
}
