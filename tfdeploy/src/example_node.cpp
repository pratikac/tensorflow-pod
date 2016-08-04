#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "tfutils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;


class tt_t
{
    private:
        high_resolution_clock::time_point start, end;

    public:
        void tic()
        {
            start = high_resolution_clock::now();
        }
        double toc()
        {
            end = high_resolution_clock::now();
            return duration_cast<duration<double>>(end - start).count();
        }
};

// get k largest elements of arg in (wt, idx)
// sigh C
void get_k_largest(int k, vector<float>& arg, vector<float>& wt, vector<int>& idx)
{
    struct comp_t
    {
        comp_t(const vector<float>& v) : _v(v) {}
        bool operator()(float a, float b) {return _v[a] > _v[b];}
        const vector<float>& _v;
    };

    idx.resize(arg.size());
    for(size_t i=0; i<idx.size(); i++)
        idx[i] = i;

    partial_sort(idx.begin(), idx.begin() + k, idx.end(), comp_t(arg));
    
    wt.clear();
    wt.resize(k);
    idx.resize(k);
    for(size_t i=0; i<idx.size(); i++)
        wt[i] = arg[idx[i]];
}

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

        tt_t tt;
        tt.tic();

        tf.run_network(x, y);
        printf("dt: %.3f [ms]\n", tt.toc()*1000);

        vector<float> tmp;
        vector<int> idx;
        get_k_largest(5, y, tmp, idx);
        for(int i=0; i<5; i++)
            printf("%d [%d, %.3f]\n", i, idx[i], tmp[i]);

    }
    return 0;
}
