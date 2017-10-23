//
// Created by 翁规范 on 10/22/17.
//

#ifndef PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_FEATURE_HPP
#define PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_FEATURE_HPP


#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dirent.h>
#include <common_tool.hpp>

using cv::Mat;
using cv::Mat_;
using cv::Point;
using cv::HOGDescriptor;
using cv::Size;
using cv::imshow;
using cv::imwrite;
using cv::imread;
using cv::Scalar;
using cv::Rect;
using cv::filter2D;
using cv::AutoBuffer;

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;

// string path_feature("D:\\pedestrian_detection_project\\pedestrian_detect_CNNs\\log\\sample_compare_feature.txt");

// ofstream log_feature(path_feature);

class pedestrian_detection_CNNs_feature {

public:

    static void init_max_map();

    int get_size() {
        return feature_size;
    }

    Size get_size_max_pool() {
        return size_max_pool;
    }

    void max_pool(Mat &src_img, int index, int offset_buf) {

        // Mat dst1(size_max_pool,src_img.type(),dbuf_1);


        int rows_dst = (int) (src_img.rows / 3), cols_dst = (int) (src_img.cols / 3);

        int ii = 0, jj = 0, k = offset_buf;

        uchar *row_0, *row_1, *row_2;

        for (int i = 0; i < rows_dst; i++) {

            row_0 = src_img.ptr<uchar>(ii);
            row_1 = src_img.ptr<uchar>(ii + 1);
            row_2 = src_img.ptr<uchar>(ii + 2);

            jj = 0;

            for (int j = 0; j < cols_dst; j++) {

                // int t;

                // a=max_map[  row_0[jj]  ]  [ row_0[jj+1] ];
                // b=max_map[  max_map[  row_0[jj]  ]  [ row_0[jj+1] ] ]  [ row_0[jj+2] ];

                // c=max_map[  row_1[jj]  ]  [ row_1[jj+1] ];
                // d=max_map[  max_map[  row_1[jj]  ]  [ row_1[jj+1] ] ]  [ row_1[jj+2] ];

                // e=max_map[  row_2[jj]  ]  [ row_2[jj+1] ];
                // f=max_map[  max_map[  row_2[jj]  ]  [ row_2[jj+1] ] ]  [ row_2[jj+2] ];

                // dbuf[k]=max_map[  max_map[  max_map[  max_map[ (int) row_0[jj]  ]  [ (int) row_0[jj+1] ] ]  [ (int) row_0[jj+2] ]  ][  max_map[  max_map[ (int) row_1[jj]  ]  [ (int) row_1[jj+1] ] ]  [ (int) row_1[jj+2] ]  ]  ][   max_map[  max_map[ (int) row_2[jj]  ]  [ (int) row_2[jj+1] ] ]  [ (int) row_2[jj+2] ]   ];
                dbuf[k] = max_map[max_map[max_map[max_map[row_0[jj]][row_0[jj + 1]]][row_0[jj +
                                                                                           2]]][max_map[max_map[row_1[jj]][row_1[
                        jj + 1]]][row_1[jj + 2]]]][max_map[max_map[row_2[jj]][row_2[jj + 1]]][row_2[jj + 2]]];

                // dbuf[k]=(float)max_map[max_map[(int)row_0[jj]][(int)row_0[jj+1]]][max_map[(int)row_1[jj]][(int)row_1[jj+1]]];

                // dbuf[k]=(uchar)t;

                // cout<<t<<" "<<(int)dbuf[k]<<" "<<(int)row_0[jj]<<endl;

                jj = jj + 3;

                k++;
            }

            ii += 3;

        }


    }

    void compute(Mat &sample_img, float *dst_array) {
        // void compute(Mat &sample_img,uchar*dst_array){

        Size sample_size(32, 64);

        // Mat img_result(sample_size.height-2,(sample_size.width-2)/2, CV_8UC1,dst_array);

        dbuf = dst_array;

        // Mat feature_mat(size_max_pool.height*2,size_max_pool.width,sample_img.type(),dbuf);

        int offset = size_max_pool.area();

        //gradient feature map
        Mat dst_img_x_grad, dst_img_y_grad;

        filter2D(sample_img, dst_img_x_grad, -1, kernelx_gradient);
        filter2D(sample_img, dst_img_y_grad, -1, kernely_gradient);

        Mat dst_img_grad = (dst_img_x_grad + dst_img_y_grad)(rect_extracted);

        max_pool(dst_img_grad, 0, 0);

        // laplace feature map
        // Mat dst_img_laplace;
        // filter2D(sample_img,dst_img_laplace,-1,kernel_laplace);
        // dst_img_laplace=dst_img_laplace(rect_extracted);

        // max_pool(dst_img_laplace,1,offset);
        // max_pool(dst_img_laplace,0,0);

        dst_img_x_grad.release();
        dst_img_y_grad.release();
        dst_img_grad.release();
        // dst_img_laplace.release();

        // if(compute_num<30){

        // int j=0;
        // for(j=0;j<50;j++){
        // log_feature<<" "<<dbuf_int[j];
        // }

        // log_feature<<endl;

        // for(j=0;j<50;j++){
        // log_feature<<" "<<dst_array[j];
        // }

        // log_feature<<endl;

        // compute_num++;
        // }

        // imshow("feature_mat",feature_mat);
        // return dst_array;
    }


    pedestrian_detection_CNNs_feature(Size _sample_img_size = Size(32, 64), int _img_channel = 3) {
        sample_img_size = _sample_img_size;
        img_channel = _img_channel;

        //size of max pool Mat
        size_max_pool = Size((sample_img_size.width - 2) / 3, (sample_img_size.height - 1) / 3);

        feature_size = size_max_pool.area();

        int kernel_side = kernelx_gradient.cols;

//		int tmp_0=kernel_side/2, tmp_1=(kernel_side/2);

        rect_extracted = Rect(kernel_side / 2, kernel_side / 2, sample_img_size.width - (kernel_side / 2) * 2,
                              sample_img_size.height - (kernel_side / 2) * 2 + 1);


        //buffer for feature
        // dbuf =(uchar*)malloc(sizeof(uchar)*feature_size);

        // dbuf_0 = (uchar*)malloc(sizeof(uchar)*size_max_pool.area());

        // dbuf_1 = (uchar*)malloc(sizeof(uchar)*size_max_pool.area());;

    }

    static int max_map[256][256];

private:


    int type_num = 2;
    int dbuf_int[100000];
    Size sample_img_size;
    int img_channel, feature_size = 0, compute_num = 0;
    Size size_max_pool;
    float *dbuf;
    Rect rect_extracted;
    // uchar* dbuf_0;
    // uchar* dbuf_1;


    Mat kernelx_gradient = (Mat_<float>(3, 3) << -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1),

            kernely_gradient = (Mat_<float>(3, 3) << -1, -2, -1,
            0, 0, 0,
            1, 2, 1),

            kernel_laplace = (Mat_<float>(3, 3) << -1, -1, -1,
            -1, 8, -1,
            -1, -1, -1);

};

int pedestrian_detection_CNNs_feature::max_map[256][256] = {{0}};

void pedestrian_detection_CNNs_feature::init_max_map() {

    // string path("D:\\Cprogram\\opencv_code\\opencv_project\\opencv_CNNs\\pedestrian_detect_CNNs\\maps\\max_map.txt");
    string path("/Users/wengguifan/CLionProjects/pedestrian_detection_CNN/map/max_map.txt");

    ifstream input(path);

    string s;

    // int tmp;

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {

            if (input >> s) {
                //                cout<<s<<endl;
                sscanf(s.c_str(), "%d", &max_map[i][j]);
                // max_map[i][j]=tmp;
            } else {
                break;
            }

        }
    }
    input.close();
}

#endif //PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_FEATURE_HPP
