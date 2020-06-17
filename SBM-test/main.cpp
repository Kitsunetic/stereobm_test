#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <algorithm>

#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/opencv.hpp>
#include "yg_imagelib.hpp"
#include "yg_stereobm.hpp"
#include "pathkit.hpp"

//#define ADJUST_WLS_FILTER
#define USE_YG_STEREOBM
#define USE_ROTATED_STEREO
#define VERBOSE_IMWRITE

using namespace std;
using namespace cv;


void adjust_stereobm(Mat &imL_, Mat &imR_, Mat &depth, Mat &depth_rot, int num_disp, int window_size, float baseline=0.5) {
#ifdef USE_ROTATED_STEREO
    // rotated stereo image pair make better result with StereoBM
    Mat imL = yg::rotate_image(imL_, Vec3d(RAD(90), 0, 0));
    Mat imR = yg::rotate_image(imR_, Vec3d(RAD(90), 0, 0));
    cv::rotate(imL, imL, cv::ROTATE_90_CLOCKWISE);
    cv::rotate(imR, imR, cv::ROTATE_90_CLOCKWISE);
#else
    Mat imL = imL_.clone();
    Mat imR = imR_.clone();
#endif
    
    // make gray
    if(imL.channels() != 1) cvtColor(imL, imL, CV_BGR2GRAY);
    if(imR.channels() != 1) cvtColor(imR, imR, CV_BGR2GRAY);
    
    Mat disp, disp_sbm;
#ifdef USE_YG_STEREOBM
    Ptr<yg::StereoBM> sbm = yg::StereoBM::create(num_disp, window_size);
#else
    Ptr<StereoBM> sbm = StereoBM::create(num_disp, window_size);
#endif

    sbm->compute(imL, imR, disp_sbm);
#ifdef VERBOSE_IMWRITE
    imwrite("disp_sbm.png", disp_sbm);
#endif
    
    disp = disp_sbm.clone();
    disp.convertTo(disp, CV_32FC1);
    disp = disp / 16.0f;
    
    float h = (float)disp.cols;
    float pi_h = (float)M_PI / h;
    Mat rl(disp.rows, disp.cols, CV_32FC1, Scalar(0));
    Mat rl_mask, rl_out, rl_out_rot;
    
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
    
    // depth
    rl_mask = rl > 0;
    rl.copyTo(rl_out, rl_mask);
    normalize(rl_out, depth, 0, 255, CV_MINMAX, CV_8U);
    cvtColor(depth, depth, CV_GRAY2BGR);
    
    // depth_rot
    cv::rotate(rl_out, rl_out_rot, ROTATE_90_COUNTERCLOCKWISE);
    rl_out_rot = yg::rotate_image(rl_out_rot, Vec3d(RAD(-90), 0, 0));
    normalize(rl_out_rot, depth_rot, 0, 255, CV_MINMAX, CV_8U);
    cvtColor(depth_rot, depth_rot, CV_GRAY2BGR);
}

void adjust_stereobm(Mat &imL, Mat &imR, Mat &depth, int num_disp, int window_size, float baseline=0.5) {
    Mat _depth_rot;
    adjust_stereobm(imL, imR, depth, _depth_rot, num_disp, window_size, baseline);
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
    int window_size = 21;
    if(argc >= 4) num_disp = atoi(argv[3]);
    if(argc >= 5) window_size = atoi(argv[4]);
    
    Mat imL = imread(pathL);
    Mat imR = imread(pathR);
    if(!imL.data || !imR.data) {
        cout << "Cannot load image" << endl;
        return 1;
    }
    
    // stereobm
    Mat depth, depth2, depth4;
    adjust_stereobm(imL, imR, depth, num_disp, window_size);
    adjust_stereobm(imL, imR, depth2, num_disp, 15);
    adjust_stereobm(imL, imR, depth4, num_disp, 9);
    
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
    imwrite("depth_comb_max.png", depth_c_max);
    imwrite("depth_comb_min.png", depth_c_min);
    imwrite("depth_comb_mean.png", depth_c_mean);
    imwrite("depth_comb.png", depth_c);
}
