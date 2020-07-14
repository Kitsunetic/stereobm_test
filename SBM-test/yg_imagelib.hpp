#include <cmath>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <algorithm>

#include <opencv2/opencv.hpp>

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

#define DEPTH_MIN -5.0f
#define DEPTH_MAX 20.0f

using namespace std;
using namespace cv;


namespace yg {

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
    vec_pixel[0] = (int)((double)(height * vec_rot[0]) / M_PI);
    vec_pixel[1] = (int)((double)(width  * vec_rot[1]) / (2*M_PI));

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
    #pragma omp parallel for
    for(int i = 0; i < im_height; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            // inverse warping
            Vec2i vec_pixel = rotate_pixel(Vec2i(i, j) 
                                         , rot_mat
                                         , im_width, im_height);
            srci.at<float>(i, j) = (float)vec_pixel[0];
            srcj.at<float>(i, j) = (float)vec_pixel[1];
        }
    }
    remap(im, im_out, srcj, srci, INTER_LINEAR);

    return im_out;
}
}
