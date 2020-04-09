#define _USE_MATH_DEFINES
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "yg_stereobm.hpp"

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

// option
#define DEBUG

#define DEPTH_MIN 1e-6
#define DEPTH_MAX 30.0f

#define INPUT_RESIZE 1.0f
#define SHOW_RESIZE 2.0f

using namespace std;
using namespace cv;

// Mouse event
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    Mat *im = reinterpret_cast<Mat*>(userdata);

    if (event == EVENT_LBUTTONDOWN)
    {
        cout << "pixel = (" << x << ", " << y << "), " << im->at<float>(y*SHOW_RESIZE, x*SHOW_RESIZE) << endl;
    }
}

// XYZ-eular rotation 
Mat eular2rot(Vec3d theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
     
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
    // Combined rotation matrix
    Mat R = R_x * R_y * R_z;
     
    return R;
}

// Rotation matrix to rotation vector in XYZ-eular order
Vec3d rot2eular(Mat R)
{
    double sy = sqrt(R.at<double>(2,2) * R.at<double>(2,2) +  R.at<double>(1,2) * R.at<double>(1,2) );
 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular)
    {
        x = atan2(-R.at<double>(1,2) , R.at<double>(2,2));
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    else
    {
        x = 0;
        y = atan2(R.at<double>(0,2), sy);
        z = atan2(-R.at<double>(0,1), R.at<double>(0,0));
    }
    return Vec3d(x, y, z);
}

// rotate pixel, in_vec as input(row, col)
Vec2i rotate_pixel(const Vec2i& in_vec, Mat& rot_mat, int width, int height)
{
    Vec2d vec_rad = Vec2d(M_PI*in_vec[0]/height, 2*M_PI*in_vec[1]/width);

    Vec3d vec_cartesian;
    vec_cartesian[0] = -sin(vec_rad[0])*cos(vec_rad[1]);
    vec_cartesian[1] = sin(vec_rad[0])*sin(vec_rad[1]);
    vec_cartesian[2] = cos(vec_rad[0]);

    double* rot_mat_data = (double*)rot_mat.data;
    Vec3d vec_cartesian_rot;
    vec_cartesian_rot[0] = rot_mat_data[0]*vec_cartesian[0] + rot_mat_data[1]*vec_cartesian[1] + rot_mat_data[2]*vec_cartesian[2];
    vec_cartesian_rot[1] = rot_mat_data[3]*vec_cartesian[0] + rot_mat_data[4]*vec_cartesian[1] + rot_mat_data[5]*vec_cartesian[2];
    vec_cartesian_rot[2] = rot_mat_data[6]*vec_cartesian[0] + rot_mat_data[7]*vec_cartesian[1] + rot_mat_data[8]*vec_cartesian[2];

    Vec2d vec_rot;
    vec_rot[0] = acos(vec_cartesian_rot[2]);
    vec_rot[1] = atan2(vec_cartesian_rot[1], -vec_cartesian_rot[0]);
    if(vec_rot[1] < 0)
        vec_rot[1] += M_PI*2;

    Vec2i vec_pixel;
    vec_pixel[0] = height*vec_rot[0]/M_PI;
    vec_pixel[1] = width*vec_rot[1]/(2*M_PI);

    return vec_pixel;
}

Mat rotate_image(Mat& im, Vec3d theta)
{
    int im_width = im.cols;
    int im_height = im.rows;
    double im_size = im_width*im_height;
    Size im_shape(im_height, im_width);

    Mat im_out(im.rows, im.cols, im.type());

    Mat srci(im_height, im_width, CV_32F);
	Mat srcj(im_height, im_width, CV_32F);

    Mat rot_mat = eular2rot(theta);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            // inverse warping
            Vec2i vec_pixel = rotate_pixel(Vec2i(i, j) 
                                         , rot_mat
                                         , im_width, im_height);
            srci.at<float>(i, j) = vec_pixel[0];
            srcj.at<float>(i, j) = vec_pixel[1];
        }
    }
    remap(im, im_out, srcj, srci, INTER_LINEAR);

    return im_out;
}

int main(int argc, char* argv[])
{
    bool is_depth_gt = false;
    int window_size, numdisp;
    string left_name, right_name;

    if(argc != 5)
    {
        cout << "usage : filename.out <left_image> <right_image> <window_size> <number_of_disparity>" << endl;
        return 0;
    }
    else
    {
        left_name = argv[1];
        right_name = argv[2];
        window_size = stoi(argv[3]);
        numdisp = stoi(argv[4]);
    }

    Mat left_image = imread(left_name);
    Mat right_image = imread(right_name);

    resize(left_image, left_image, Size(), 1/INPUT_RESIZE, 1/INPUT_RESIZE);
    resize(right_image, right_image, Size(), 1/INPUT_RESIZE, 1/INPUT_RESIZE);

    Mat left_rotate = rotate_image(left_image, Vec3d(RAD(90), 0, 0));
    Mat right_rotate = rotate_image(right_image, Vec3d(RAD(90), 0, 0));
    cv::rotate(left_rotate, left_rotate, cv::ROTATE_90_CLOCKWISE);
    cv::rotate(right_rotate, right_rotate, cv::ROTATE_90_CLOCKWISE);

    Mat left_gray, right_gray;
    cvtColor(left_rotate, left_gray, CV_RGB2GRAY);
    cvtColor(right_rotate, right_gray, CV_RGB2GRAY);

    Mat disp;
    cout << "use yg::BM" << endl;
    Ptr<yg::StereoBM> sbm = yg::StereoBM::create(numdisp, window_size);
    Mat disp_sbm;
    sbm->compute(left_gray, right_gray, disp_sbm);
    disp = disp_sbm.clone();
    disp.convertTo(disp, CV_32FC1);
    disp = disp/16.0f;

    Mat disp2;
    Mat left_gray_invert, right_gray_invert;
    cv::rotate(left_gray, left_gray_invert, cv::ROTATE_180);
    cv::rotate(right_gray, right_gray_invert, cv::ROTATE_180);
    cout << "use yg::BM" << endl;
    Mat disp_sbm2;
    sbm->compute(right_gray_invert, left_gray_invert, disp_sbm2);
    disp2 = disp_sbm2.clone();
    disp2.convertTo(disp2, CV_32FC1);
    disp2 = disp2/16.0f;
    cv::rotate(disp2, disp2, cv::ROTATE_180);

    Mat rl(disp.rows, disp.cols, CV_32FC1, Scalar(0));
    Mat rl2(disp.rows, disp.cols, CV_32FC1, Scalar(0));
    float baseline = 0.5;
    float h = disp.cols;
    float pi_h = M_PI / h;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < disp.rows; i++)
    {
        for(int j = 0; j < disp.cols; j++)
        {
            float d = disp.at<float>(i, j);
            float d2 = disp2.at<float>(i, j);
            if(d > DEPTH_MIN)
            {
                float val = baseline/((sin(j*pi_h)/tan((j - d)*pi_h)) - cos(j*pi_h));
                rl.at<float>(i, j) = std::min(val, DEPTH_MAX);
                float val2 = baseline/((sin(j*pi_h)/tan((j - d2)*pi_h)) - cos(j*pi_h));
                rl2.at<float>(i, j) = std::min(val2, DEPTH_MAX);
            }
        }
    }

    Mat rl_out, rl_out_rotated;
    Mat rl_mask = rl > 0;
    rl.copyTo(rl_out, rl_mask);

    cv::rotate(rl_out, rl_out_rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    rl_out_rotated = rotate_image(rl_out_rotated, Vec3d(RAD(-90), 0, 0));

    Mat tmp;
    normalize(rl_out_rotated, tmp, 0, 255, CV_MINMAX, CV_8U);
    cvtColor(tmp, tmp, CV_GRAY2RGB);
    imwrite("output.png", tmp);

    Mat rl_out2, rl_out_rotated2;
    Mat rl_mask2 = rl2 > 0;
    rl2.copyTo(rl_out2, rl_mask2);

    cv::rotate(rl_out2, rl_out_rotated2, cv::ROTATE_90_COUNTERCLOCKWISE);
    rl_out_rotated2 = rotate_image(rl_out_rotated2, Vec3d(RAD(-90), 0, 0));

    Mat tmp2;
    normalize(rl_out_rotated2, tmp2, 0, 255, CV_MINMAX, CV_8U);
    cvtColor(tmp2, tmp2, CV_GRAY2RGB);
    imwrite("output2.png", tmp2);

#ifdef DEBUG
    cout << "debug enabled" << endl;
    Mat plus_region = rl > 0;
    Mat minus_region = rl < 0;
    Mat zero_region = rl == 0;
    
    vector<Mat> merge_array;
    Mat merge_mat;
    merge_array.push_back(plus_region);
    merge_array.push_back(minus_region);
    merge_array.push_back(zero_region);
    merge(merge_array, merge_mat);

    imwrite("plus_minus_zero.png", merge_mat);
    imwrite("left_rotated.png", left_rotate);
    imwrite("right_rotated.png", right_rotate);

    Mat disp_rot;
    cv::rotate(disp_sbm, disp_rot, cv::ROTATE_90_COUNTERCLOCKWISE);
    disp_rot = rotate_image(disp_rot, Vec3d(RAD(-90), 0, 0));
    imwrite("disp.png", disp_rot);
    
    vector<Mat> left_right_merge;
    Mat left_right_merge_mat;
    left_right_merge.push_back(left_gray);
    left_right_merge.push_back(right_gray);
    left_right_merge.push_back(Mat::zeros(left_gray.rows, left_gray.cols, CV_8UC1));
    merge(left_right_merge, left_right_merge_mat);
    imwrite("left_right_merge.png", left_right_merge_mat);

#endif
}