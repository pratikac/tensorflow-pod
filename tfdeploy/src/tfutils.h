#ifndef __tfdeploy_utils_h__
#define __tfdeploy_utils_h__

#include <cstdio>
#include <cstdlib>
#include "tensor_c_api.h"

#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;

class tfdeploy_t
{
    public:
        TF_SessionOptions* session_opt;
        TF_Status* status;
        TF_Session* session;

        typedef struct
        {
            char* fname;

            char* input_name;
            char* output_name;

            int64_t xdim[4];
            int64_t ydim[4];

            int64_t input_size;
            int64_t output_size;

            uint8_t input_mean[3];
            uint8_t input_std[3];
        } network_opt_t;

        network_opt_t network_opt;

        tfdeploy_t()
        {
            session_opt = TF_NewSessionOptions();
            status = TF_NewStatus();
            session = TF_NewSession(session_opt, status);
        }

        ~tfdeploy_t()
        {
            TF_CloseSession(session, status);
            TF_DeleteSession(session, status);
        }

        int init_network_opts(
                char* fname,
                char* input_name,
                int64_t xdim[4],
                uint8_t xmean[3], uint8_t xstd[3],
                char* output_name,
                int64_t ydim[4])
        {
            auto& o = network_opt;
            o.input_name = strdup(input_name);
            o.output_name = strdup(output_name);
            for(int i=0; i<4; i++)
            {
                o.xdim[i] = xdim[i];
                o.ydim[i] = ydim[i];
            }
            for(int i=0; i<3; i++)
            {
                o.input_mean[i] = xmean[i];
                o.input_std[i] = xstd[i];
            }

            o.input_size = o.xdim[1]*o.xdim[2]*o.xdim[3];
            o.output_size = o.ydim[1]*o.ydim[2]*o.ydim[3];

            o.fname = strdup(fname);
            read_protobuf(o.fname);
        }

        static void tf_dealloc(void* data, size_t len, void* arg){}
        
        TF_Tensor* tf_alloc(const int64_t* dims, int num_dims,
                void* data, size_t len)
        {
            return TF_NewTensor(TF_FLOAT, dims, num_dims, data, sizeof(TF_FLOAT)*len,
                    tfdeploy_t::tf_dealloc, 0);
        }

        bool check_status(TF_Status* s)
        {
            if(TF_GetCode(s) == TF_Code::TF_OK)
                return true;
            else
            {
                printf("status: %d\n", TF_GetCode(s));
                exit(1);
            }
            return false;
        }

        int read_protobuf(char* fname)
        {
            FILE* fp = fopen(fname, "r");
            fseek(fp, 0, SEEK_END);
            unsigned long fsize = ftell(fp);
            rewind(fp);

            char *pb = (char*) malloc(fsize + 1);
            fread(pb, sizeof(char), fsize, fp);
            fclose(fp);
            pb[fsize] = 0;

            // assumes a new graph was given
            TF_ExtendGraph(session, pb, fsize, status);
            if(check_status(status))
                printf("Read protobuf: %s\n", fname);
            else
                return 1;

            free(pb);
            return 0;
        }

        // assume 3-channel RBG images
        int run_network(vector<cv::Mat>& X, vector<float>& Y)
        {
            auto& opt = network_opt;

            for(auto& i : X)
            {
                assert( i.cols == opt.xdim[1] &&
                        i.rows == opt.xdim[2] &&
                        i.channels() == opt.xdim[3]);

                i.convertTo(i, CV_32FC3, 1.0);
                subtract(i, 128, i);
                divide(i, 128, i);
            }

            size_t batch_size = X.size();
            opt.xdim[0] = batch_size;
            
            // copy images into tensor
            size_t Nx = batch_size*opt.input_size;
            size_t Ny = batch_size*opt.output_size;
            float* xbuffer = new float[Nx];
            
            for(size_t i=0; i<X.size(); i++)
                memcpy(xbuffer, X[i].data, sizeof(float)*opt.input_size);
            
            auto x = tf_alloc(opt.xdim, 4, xbuffer, Nx);

            TF_Tensor* tfx[1] = {x};
            TF_Tensor* tfy[1];

            const char* input_names[] = {opt.input_name};
            const char* output_names[] = {opt.output_name};

            TF_Run(session, NULL,
                    input_names, tfx, 1,
                    output_names, tfy, 1,
                    NULL, 0, NULL,
                    status);

            Y.clear();
            Y.resize(Ny);
            memcpy(&Y[0], TF_TensorData(tfy[0]), Ny*sizeof(float));
        }
};



#endif
