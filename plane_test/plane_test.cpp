#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include "yg_stereobm.hpp"

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

// option
#define DEPTH_MIN 1e-6
#define DEPTH_MAX 30.0f

#define INPUT_RESIZE 2.0f
#define SHOW_RESIZE 2.0f

#define WINDOW_SIZE 7

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    Mat left = imread("../cones/im2.ppm");
    Mat right = imread("../cones/im6.ppm");
    Mat left_gray, right_gray;
    cvtColor(left, left_gray, CV_BGR2GRAY);
    cvtColor(right, right_gray, CV_BGR2GRAY);

    cout << "use BM" << endl;
    Ptr<yg::StereoBM> sbm = yg::StereoBM::create(128, WINDOW_SIZE);
    Mat disp_sbm;
    sbm->compute(left_gray, right_gray, disp_sbm);
    
    Mat disp = disp_sbm;
    disp.convertTo(disp, CV_32FC1);
    disp = disp/ 16.0f;

    Mat tmp;
    normalize(disp, tmp, 0, 255, CV_MINMAX, CV_8U);
    cvtColor(tmp, tmp, CV_GRAY2RGB);
    imwrite("output.png", tmp);
}