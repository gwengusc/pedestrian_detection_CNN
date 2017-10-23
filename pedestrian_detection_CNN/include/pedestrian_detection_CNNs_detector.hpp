//
// Created by 翁规范 on 10/22/17.
//

#ifndef PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_DETECTOR_HPP
#define PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_DETECTOR_HPP


#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <dirent.h>
#include <algorithm>
#include <common_tool.hpp>
// #include <pedestrian_detection_CNNs_feature_170525_0.hpp>
// #include <pedestrian_detection_CNNs_feature_170622_1.hpp>
#include <pedestrian_detection_CNNs_feature.hpp>


using cv::Mat;
using cv::Point;
using cv::HOGDescriptor;
using cv::Size;
using cv::imshow;
using cv::imwrite;
using cv::imread;
using cv::Scalar;
using cv::Rect;
using cv::namedWindow;
using cv::setMouseCallback;

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;

class pedestrian_detection_CNNs_detector {
private:

    Size feature_size;
    int (*max_map_ptr)[256];
    CvANN_MLP *cnn_pointer;//��������
    Mat kernelx_gradient = (Mat_<float>(3, 3) << -1, 0, 1,
            -2, 0, 2,
            -1, 0, 1),

            kernely_gradient = (Mat_<float>(3, 3) << -1, -2, -1,
            0, 0, 0,
            1, 2, 1);

public:

    pedestrian_detection_CNNs_detector(Size _feature_size) {
        feature_size = _feature_size;
        max_map_ptr = pedestrian_detection_CNNs_feature::max_map;
    }


    vector<Rect> pre_detect(vector<Rect> &targets, Mat &src_img, int step_size, float ratio) {

        int feature_height = feature_size.height, feature_width = feature_size.width;

        int rows = (src_img.rows - 1) / 3, cols = (src_img.cols - 1) / 3;

        int count_cols_step = (cols - feature_width + step_size) / step_size;
        int count_rows_step = (rows - feature_height + step_size) / step_size;


        if (count_cols_step < 1 || count_rows_step < 1) {
            return targets;
        }

        rows = count_rows_step * step_size;
        cols = count_cols_step * step_size;


        Mat dst_sobel_x, dst_sobel_y;

        filter2D(src_img, dst_sobel_x, -1, kernelx_gradient);
        filter2D(src_img, dst_sobel_y, -1, kernely_gradient);


        Mat dst_sobel_CV_8U = dst_sobel_x + dst_sobel_y;

        Mat dst_max_pool = max_pool(dst_sobel_CV_8U);

        int i, j, k = 0, ii, jj;

        Mat feature_mat(1, feature_size.area(), CV_32FC1), response_mat;

        float *feature_data = feature_mat.ptr<float>(0);
        float *response_result;

        float *ptr;


        int rect_num = 0;

        for (i = 0; i < rows; i += step_size) {

            for (j = 0; j < cols; j += step_size) {

                k = 0;

                rect_num++;

                for (ii = i; ii < feature_height + i; ii++) {

                    ptr = dst_max_pool.ptr<float>(ii);

                    for (jj = j; jj < feature_width + j; jj++) {

                        feature_data[k] = ptr[jj];

                        k++;
                    }

                }


                cnn_pointer->predict(feature_mat, response_mat);

                response_result = response_mat.ptr<float>(0);

                if (response_result[0] > response_result[1]) {

                    targets.push_back(Rect((int) ((j * 3 - 1) / ratio), (int) ((i * 3 - 1) / ratio),
                                           (int) ((feature_width * 3 + 1) / ratio),
                                           (int) ((feature_height * 3 + 1) / ratio)));
                    // targets.push_back( Rect( (int)((ii*3-1)/ratio),  (int)( (jj*3-1)/ratio ),  (int)((feature_width*3+1)/ratio),  (int) ((feature_height*3+1)/ratio) )  ); // it is not ii and jj. They are i and j;
                    // the constructor of Rect is Rect(x(col),y(row),width,height);
                }

            }

        }


        return targets;
        // return post_process(targets);
    }

    vector<Rect> post_process(vector<Rect> &targets, vector<Rect> &result_processed, int row, int col, float ratio) {

        int i, j, results_size = targets.size();
        Rect rect0, rect_intercect;

        for (i = 0; i < results_size; i++) {

            rect0 = targets[i];


            for (j = i + 1; j < results_size; j++) {

                rect_intercect = rect0 & targets[j];

                if ((rect_intercect.area() > rect0.area() * 0.9)) {
                    break;
                }

            }

            if (j == results_size) {


                if (rect0.x < 0) {
                    rect0.x = 0;
                }

                if (rect0.y < 0) {
                    rect0.y = 0;
                }

                if (rect0.x + rect0.width >= col - 1) {

                    rect0.width = col - 1 - rect0.x;


                }

                if (rect0.y + rect0.height >= row - 1) {
                    rect0.height = row - 1 - rect0.y;
                }


                result_processed.push_back(rect0);
            }
        }

        return result_processed;
    }


    // A simple post-process (NMS, non-maximal suppression)
    // "result" -- rectangles before merging
    //          -- after this function it contains rectangles after NMS
    // "combine_min" -- threshold of how many detection are needed to survive
    void post_process_NMS(std::vector<Rect> &result, const int combine_min, float overlap_proportion) {
        std::vector<Rect> res1;
        std::vector<Rect> resmax;
        std::vector<int> res2;
        bool yet;
        Rect rectInter;

        for (unsigned int i = 0, size_i = result.size(); i < size_i; i++) {
            yet = false;
            Rect result_i = result[i];
            for (unsigned int j = 0, size_r = res1.size(); j < size_r; j++) {
                Rect resmax_j = resmax[j];
                rectInter = result_i & resmax[j];

                if (rectInter.area() > overlap_proportion * result_i.area() &&
                    rectInter.area() > overlap_proportion * resmax_j.area()) {
                    Rect res1_j = res1[j];
                    // resmax_j.Union(resmax_j,result_i);
                    resmax_j = resmax_j | result_i;

                    res1_j.x = result_i.x;
                    res1_j.y = result_i.y;
                    res1_j.height += result_i.height;
                    res1_j.width += result_i.width;


                    res2[j]++;

                    yet = true;

                    break;
                }
            }

            if (yet == false) {
                res1.push_back(result_i);
                resmax.push_back(result_i);
                res2.push_back(1);
            }
        }

        for (unsigned int i = 0, size = res1.size(); i < size; i++) {
            const int count = res2[i];
            Rect res1_i = res1[i];
            res1_i.x /= count;
            res1_i.y /= count;
            res1_i.height /= count;
            res1_i.width /= count;
        }

        result.clear();
        for (unsigned int i = 0, size = res1.size(); i < size; i++)
            if (res2[i] > combine_min)
                result.push_back(res1[i]);
    }

    Mat max_pool(Mat &src_img) {

        int rows_dst = (int) ((src_img.rows - 1) / 3), cols_dst = (int) ((src_img.cols - 1) / 3);

        float *dbuf = (float *) malloc(sizeof(float) * rows_dst * cols_dst);

        Mat dst_max_pool = Mat(rows_dst, cols_dst, CV_32FC1, dbuf);
        // uchar* dbuf=(uchar*)malloc(sizeof(uchar)*rows_dst*cols_dst);

        // Mat dst_max_pool= Mat( rows_dst, cols_dst, CV_8U, dbuf);

        int ii = 1, jj, index_dbuf = 0;

        uchar *row_0, *row_1, *row_2;

        for (int i = 0; i < rows_dst; i++) {

            row_0 = src_img.ptr<uchar>(ii);
            row_1 = src_img.ptr<uchar>(ii + 1);
            row_2 = src_img.ptr<uchar>(ii + 2);

            jj = 1;

            for (int j = 0; j < cols_dst; j++) {

                dbuf[index_dbuf] = (float) max_map_ptr[max_map_ptr[max_map_ptr[max_map_ptr[row_0[jj]][row_0[jj +
                                                                                                            1]]][row_0[
                        jj + 2]]][max_map_ptr[max_map_ptr[row_1[jj]][row_1[jj + 1]]][row_1[jj +
                                                                                           2]]]][max_map_ptr[max_map_ptr[row_2[jj]][row_2[
                        jj + 1]]][row_2[jj + 2]]];

                jj = jj + 3;

                index_dbuf++;

            }

            ii += 3;

        }

        return dst_max_pool;
    }

    vector<Rect> detect(Mat &src_img, int step_size, string model_cnn_path) {

        CvANN_MLP bp;
        bp.load(model_cnn_path.c_str());
        cnn_pointer = &bp;
        vector<Rect> targets;
        pre_detect(targets, src_img, step_size, 1.0);
        // vector<Rect> results;
        // results.clear();
        // post_process_NMS(targets,2);
        // post_process(targets,results,src_img.size(),1.0);

        // return results;
        return targets;
    }

    vector<Rect> detect_multi(Mat &src_img, int step_size, float ratio_step, string model_cnn_path) {

        Mat src_img_copy;
        src_img.copyTo(src_img_copy);


        int org_width = src_img.cols, org_height = src_img.rows, feature_width = feature_size.width, feature_height = feature_size.height;

        int width = org_width, height = org_height;

        CvANN_MLP bp;
        bp.load(model_cnn_path.c_str());
        cnn_pointer = &bp;

        vector<Rect> results;
        results.clear();

        vector<Rect> targets;
        targets.clear();

        float ratio = 1.0;


        while (width >= feature_width && height >= feature_height) {

            resize(src_img_copy, src_img_copy, Size(width, height), 0, 0, CV_INTER_AREA);

            pre_detect(targets, src_img_copy, step_size, ratio);

            ratio = ratio * ratio_step;
            height =
                    ratio * org_height;//pay attention! remember to use the original and fixed value as the source value
            width = ratio * org_width;
            //width=ratio*width; spent several hours for debug, width changed in every loop

        }

        cout << "targets size:" << targets.size() << endl;

        post_process_NMS(targets, 2, 0.6);

        // cout<<"targets_size:"<<targets.size()<<endl;

        post_process_NMS(targets, 0, 0.6);

        post_process(targets, results, src_img.rows, src_img.cols, ratio);

        // cout<<"results size:"<<results.size()<<endl;
        targets.clear();

        return results;

    }

};

#endif //PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_DETECTOR_HPP
