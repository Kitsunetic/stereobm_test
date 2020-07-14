#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <algorithm>
#include <time.h>

#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/opencv.hpp>

#include "yg_imagelib.hpp"
#include "yg_stereobm.hpp"
#include "pathkit.hpp"
#include "debug_print.h"

// =====================================
// @@ Choose One Algorithm @@
#define USE_YG_SBM
//#define USE_CV_SBM
//#define USE_CV_SGBM
// =====================================

#define USE_ROTATED_STEREO
#define VERBOSE_IMWRITE
#define RESIZE 1.0f
#define BASELINE 0.2f
#define DEPTH_YG

#define WLS_Filter
#define WLS_LAMBDA 20000
#define WLS_SIGMA 1.2

using namespace std;
using namespace cv;


void adjust_stereobm(Mat &imL_, Mat &imR_, Mat &h_dis, Mat &v_dis, Mat &depth, Mat &depth_rot, int num_disp, int window_size, float baseline=BASELINE) {
#ifdef USE_ROTATED_STEREO
    // rotated stereo image pair make better result with StereoBM
    Mat imL = yg::rotate_image(imL_, Vec3d(RAD(89.999), 0, 0));
    Mat imR = yg::rotate_image(imR_, Vec3d(RAD(89.999), 0, 0));
    cv::rotate(imL, imL, cv::ROTATE_90_CLOCKWISE);
    cv::rotate(imR, imR, cv::ROTATE_90_CLOCKWISE);
    imwrite("rotatedl.png",imL);
    imwrite("rotatedr.png",imR);

    vector<Mat> res;
    Mat imL_g, imR_g, merged;
    cv::cvtColor(imL, imL_g, CV_BGR2GRAY);
    cv::cvtColor(imR, imR_g, CV_BGR2GRAY);
    res.push_back(imL_g);
    res.push_back(imL_g);
    res.push_back(imR_g);
    cv::merge(res, merged);
    imwrite("vertical_merged.png",merged);
#else
    Mat imL = imL_.clone();
    Mat imR = imR_.clone();
#endif
    
    // make gray
    if(imL.channels() != 1) cvtColor(imL, imL, CV_BGR2GRAY);
    if(imR.channels() != 1) cvtColor(imR, imR, CV_BGR2GRAY);
    
    Mat disp, disp_sbm;
	int maxdisp = 256;
#if defined(USE_YG_SBM)
    Ptr<yg::StereoBM> sbm = yg::StereoBM::create(num_disp, window_size);
#elif defined(USE_CV_SBM)
    Ptr<StereoBM> sbm = StereoBM::create(num_disp, window_size);
#elif defined(USE_CV_SGBM)
    Ptr<StereoSGBM> sgbm = cv::StereoSGBM::create(0, maxdisp, window_size, 0, 0, 1, 15, 0, 2, 63);
#endif

    int borderType = BORDER_CONSTANT;
    cv::Rect dispROI(maxdisp, 0, imL.cols, imL.rows);

    Mat imL_rep = imL.clone();
    Mat imR_rep = imR.clone();
    cv::copyMakeBorder(imL_rep, imL_rep, 0, 0, maxdisp, 0,CV_HAL_BORDER_CONSTANT);
    cv::copyMakeBorder(imR_rep, imR_rep, 0, 0, maxdisp, 0,CV_HAL_BORDER_CONSTANT);

#ifdef VERBOSE_IMWRITE
    imwrite("lclone.png", imL_rep);
    imwrite("rclone.png", imR_rep);
#endif
    
    // SBM or SGBM
#if defined(USE_YG_SBM) || defined(USE_CV_SBM)
    sbm->compute(imL_rep, imR_rep, disp_sbm);
#elif defined(USE_CV_SGBM)
    sgbm->compute(imL_rep, imR_rep, disp_sbm);
#endif
    // Clip black line in left side
    disp_sbm = disp_sbm(dispROI);

#ifdef VERBOSE_IMWRITE
    imwrite("disp_sbm.png", disp_sbm);
    Mat disp_sbm_rotated;
    cv::rotate(disp_sbm, disp_sbm_rotated, cv::ROTATE_90_COUNTERCLOCKWISE);         // added
    disp_sbm_rotated = yg::rotate_image(disp_sbm_rotated, Vec3d(RAD(-89.999), 0, 0));
    imwrite("sbm_rotated.png",disp_sbm_rotated);
#endif
    
    disp = disp_sbm.clone();
    disp.convertTo(disp, CV_32FC1);
    disp = disp / 16.0f;

    Mat disp_rep = disp.clone();
    cv::rotate(disp_rep, disp_rep, cv::ROTATE_90_COUNTERCLOCKWISE);
    disp_rep = yg::rotate_image(disp_rep, Vec3d(RAD(-89.999), 0, 0));
    h_dis = disp_rep.clone();
    v_dis = disp.clone();
    //imwrite("./horizontal_disparity.png",disp_rep);
    //imwrite("./vertical_disparity.png",disp);

    
    float h = (float)disp.cols;
    float pi_h = (float)M_PI / h;
    Mat rl(disp.rows, disp.cols, CV_32FC1, Scalar(0));
    
#ifdef DEPTH_YG
    #pragma omp parallel for
    for(int i = 0; i < disp.rows; i++) {
        for(int j = 0; j < disp.cols; j++) {
            float d = disp.at<float>(i, j);
            if(d > DEPTH_MIN) {
                float val = baseline / ((sin(j*pi_h)/tan((j-d)*pi_h)) - cos(j*pi_h));
                rl.at<float>(i, j) = std::min(val, DEPTH_MAX);
            }
        }
    }
#else
    #pragma omp parallel for
    for(int i=0; i<disp.rows; i++) {
        for(int j=0; j<disp.cols; j++) {
            float d = disp.at<float>(i,j);
            float val = baseline / ((sin(j*pi_h)/tan((j-d)*pi_h)) - cos(j*pi_h));
            //std::cout << val << std::endl;
            if(val > DEPTH_MAX)
                rl.at<float>(i,j) = DEPTH_MAX;
            else if((j == d) || (d <= 0))
                rl.at<float>(i,j) = DEPTH_MAX;
            else
                rl.at<float>(i,j) = val;
        }
    }
#endif
    
#ifdef VERBOSE_IMWRITE
    // [TEST]
    imwrite("[TEST] dispmap.png", disp);
    imwrite("[TEST] rl.png", rl);
#endif
    
    // masking
    Mat rl_mask, rl_out;
    rl_mask = rl > 0;
    rl.copyTo(rl_out, rl_mask);
    
#ifdef USE_ROTATED_STEREO
#ifdef VERBOSE_IMWRITE
    //imwrite("rl_out_rot.png", rl_out);
#endif
    //cv::rotate(rl_out, rl_out, cv::ROTATE_90_COUNTERCLOCKWISE);
    //rl_out = yg::rotate_image(rl_out, Vec3d(RAD(-90), 0, 0));
    cv::rotate(rl, rl, cv::ROTATE_90_COUNTERCLOCKWISE);
    rl = yg::rotate_image(rl, Vec3d(RAD(-89.999), 0, 0));
#endif
    
    // depth
    //normalize(rl_out, depth, 0, 255, CV_MINMAX, CV_8U);
    //depth = rl_out/DEPTH_MAX*255.0;
    cout << rl.type() << endl;
    imwrite("depth_no-norm.png", rl);
    depth = rl/DEPTH_MAX*255.0;
    depth.convertTo(depth, CV_8U);
    cvtColor(depth, depth, CV_GRAY2BGR);
}

void adjust_stereobm(Mat &imL, Mat &imR, Mat &h_dis, Mat &v_dis, Mat &depth, int num_disp, int window_size, float baseline=BASELINE) {
    Mat _depth_rot;
    adjust_stereobm(imL, imR, h_dis, v_dis, depth, _depth_rot, num_disp, window_size, baseline);
}

Mat adjust_WLS_filter(Mat im, Mat depth, double lambda=20000, double sigma=1.2) {
    Mat im_in, depth_in, depth_out;
    Ptr<ximgproc::DisparityWLSFilter> filter;
    
    // preprocessing: color map
    switch(im.channels()) {
        case 3: cvtColor(im, im_in, COLOR_BGR2GRAY); break;
        case 4: cvtColor(im, im_in, COLOR_BGRA2GRAY); break;
        default: im_in = im;
    }
    switch(depth.channels()) {
        case 3: cvtColor(depth, depth_in, COLOR_BGR2GRAY); break;
        case 4: cvtColor(depth, depth_in, COLOR_BGRA2GRAY); break;
        default: depth_in = depth;
    }
    
    // adjust WLS filter
    filter = ximgproc::createDisparityWLSFilterGeneric(false);
    filter->setLambda(lambda);
    filter->setSigmaColor(sigma);
    filter->filter(im_in, depth_in, depth_out);
    
    return depth_out;
}

int main(int argc, char* argv[]) {
    if(argc < 3) {
        cout << "Usage: " << argv[0] << " [left image path] [right image path]" << endl;
        return 0;
    }
    
    string pathL = argv[1];
    string pathR = argv[2];
    //string pathL = "mpeg-class-left.png";
    //string pathR = "mpeg-class-right.png";
    int num_disp = 256;
    int window_size = 17;                           // INITIAL 21, 15, 9 & RESIZED : 17,11,5
    if(argc >= 4) num_disp = atoi(argv[3]);
    if(argc >= 5) window_size = atoi(argv[4]);

    int start;
    start = clock();        // check code running time
    
    Mat imL = imread(pathL);
    Mat imR = imread(pathR);
    if(!imL.data || !imR.data) {
        cout << "Cannot load image" << endl;
        return 1;
    }
    
    // image resize
    if(RESIZE != 1.0f) {
        resize(imL, imL, Size(), 1/RESIZE, 1/RESIZE);
        resize(imR, imR, Size(), 1/RESIZE, 1/RESIZE);
    }
    
    // stereobm
    Mat depth, depth2, depth4;
    Mat h_dis, h_dis2, h_dis4, v_dis, v_dis2, v_dis4;
    START_TIME(__SBM_1__);
    adjust_stereobm(imL, imR, h_dis, v_dis, depth, num_disp, window_size, BASELINE);
    STOP_TIME(__SBM_1__);
    
    START_TIME(__SBM_2__);
    adjust_stereobm(imL, imR, h_dis2, v_dis2, depth2, num_disp, 11, BASELINE);
    STOP_TIME(__SBM_2__);
    
    START_TIME(__SBM_3__);
    adjust_stereobm(imL, imR, h_dis4, v_dis4, depth4, num_disp, 5, BASELINE);
    STOP_TIME(__SBM_3__);
    
    //save each disparity
    imwrite("h_disp.png",h_dis);
    imwrite("h_disp2.png",h_dis2);
    imwrite("h_disp4.png",h_dis4);
    imwrite("v_disp.png",v_dis);
    imwrite("v_disp2.png",v_dis2);
    imwrite("v_disp4.png",v_dis4);
    
    // save each depths
    imwrite("depth.png", depth);
    imwrite("depth2.png", depth2);
    imwrite("depth4.png", depth4);
    
    // combine depth
    Mat depth_c_max(depth.rows, depth.cols, CV_8UC1, Scalar(0));
    Mat depth_c_min(depth.rows, depth.cols, CV_8UC1, Scalar(0));
    Mat depth_c_mean(depth.rows, depth.cols, CV_8UC1, Scalar(0));
    Mat depth_c(depth.rows, depth.cols, CV_8UC1, Scalar(0));
    #pragma omp parallel for
    for(int i = 0; i < depth.rows; i++) {
        for(int j = 0; j < depth.cols; j++) {
            depth_c_max.at<uchar>(i, j) = std::max(std::max(depth.at<Vec3b>(i, j)[0], depth2.at<Vec3b>(i, j)[0]), depth4.at<Vec3b>(i, j)[0]);
            depth_c_min.at<uchar>(i, j) = std::min(std::min(depth.at<Vec3b>(i, j)[0], depth2.at<Vec3b>(i, j)[0]), depth4.at<Vec3b>(i, j)[0]);
            depth_c_mean.at<uchar>(i, j) = (uchar)(((int)depth.at<Vec3b>(i, j)[0] + (int)depth2.at<Vec3b>(i, j)[0] + (int)depth4.at<Vec3b>(i, j)[0]) / 3);
            if(depth.at<Vec3b>(i, j)[0] != 0) {
                depth_c.at<uchar>(i, j) = depth.at<Vec3b>(i, j)[0];
            } else if(depth2.at<Vec3b>(i, j)[0] != 0) {
                depth_c.at<uchar>(i, j) = depth2.at<Vec3b>(i, j)[0];
            } else {
                depth_c.at<uchar>(i, j) = depth4.at<Vec3b>(i, j)[0];
            }
        }
    }
    printf("Code elapse time : %lf\n",(double)(clock() - start)/CLOCKS_PER_SEC);

    imwrite("depth_comb_max.png", depth_c_max);
    imwrite("depth_comb_min.png", depth_c_min);
    imwrite("depth_comb_mean.png", depth_c_mean);
    imwrite("depth_comb.png", depth_c);
    
#ifdef WLS_Filter
    imL = imread(pathL);
    Mat depth_wls = adjust_WLS_filter(imL, depth_c, WLS_LAMBDA, WLS_SIGMA);
    imwrite("depth_comb_wls.png", depth_wls);
#endif
}
