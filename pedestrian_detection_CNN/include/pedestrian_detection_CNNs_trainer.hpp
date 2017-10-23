//
// Created by 翁规范 on 10/22/17.
//

#ifndef PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_TRAINER_HPP
#define PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_TRAINER_HPP


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
// #include <pedestrian_detection_CNNs_feature_170630_3.hpp>
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

class pedestrian_detection_CNNs_trainer {

private:

    int class_num = 2, sample_num_offset = 0, time = 0;

    pedestrian_detection_CNNs_feature feature;

    Mat sample_feature_mat, sample_label_mat, layer_sizes;

    string version_CNNs, model_CNNs_path;

    float *dbuf_feature, *dbuf_label;

public:

    float class_pos[2] = {1.0, 0.0}, class_neg[2] = {0.0, 1.0};

    void release_train_data() {
        sample_feature_mat.release();
        sample_label_mat.release();
        sample_num_offset = 0;
    }

    void add_samples(vector<string> samples, float *classes) {

        ofstream log_trainer;

        if (sample_num_offset == 0) {

//            string path_trainer(
//                    "D:/Cprogram/opencv_code/opencv_project/opencv_CNNs/pedestrian_detect_CNNs/log/sample_compare_trainer.txt");
//            log_trainer.open(path_trainer, std::ios::binary);
        }
        // string path_feature("D:/Cprogram/opencv_code/opencv_project/opencv_CNNs/pedestrian_detect_CNNs/log/sample_compare_feature.txt");

        // log_feature.open(path_feature,ios_base::binary);

        Mat src_img;//��ȡͼƬ

        int samples_size = samples.size(), i, index = 0, feature_size = feature.get_size();

        index = sample_num_offset * feature_size;

//		float *p;

        for (i = 0; i < samples_size; ++i) {

            src_img = imread(samples[i], CV_LOAD_IMAGE_GRAYSCALE);

            feature.compute(src_img, sample_feature_mat.ptr<float>(i + sample_num_offset));
            // feature.compute(src_img,&dbuf_feature[index]);

            // if(i<30&&sample_num_offset==0){

            // p=sample_feature_mat.ptr<float>(i+sample_num_offset);
            // // p=&dbuf_feature[index];

            // for(int j=0;j<50;j++){
            // log_trainer<<" "<<p[j];
            // }

            // log_trainer<<endl;
            // }

            // sample_label_mat.at<float>(i+sample_num_offset,0) = class;

            sample_label_mat.ptr<float>(i + sample_num_offset)[0] = classes[0];
            sample_label_mat.ptr<float>(i + sample_num_offset)[1] = classes[1];

            src_img.release();

            index += feature_size;
        }


        if (sample_num_offset == 0) {
            log_trainer.close();
        }

        sample_num_offset += i;
//		log_feature.close();
    }

    void construct_train_data(vector<string> sample_files_pos, vector<string> sample_files_neg) {

        int len_pos = sample_files_pos.size(), len_neg = sample_files_neg.size();

        int sample_num = len_pos + len_neg;

        // dbuf_feature=(float*)malloc(sizeof(float)*sample_num*feature.get_size());
        // dbuf_label=(float*)malloc(sizeof(float)*sample_num*class_num);

        // sample_feature_mat= Mat(sample_num,feature.get_size(), CV_32FC1,dbuf_feature);
        // sample_label_mat= Mat(sample_num,class_num, CV_32FC1,dbuf_label);

        sample_feature_mat = Mat(sample_num, feature.get_size(), CV_32FC1);
        sample_label_mat = Mat(sample_num, class_num, CV_32FC1);

        add_samples(sample_files_pos, class_pos);
        add_samples(sample_files_neg, class_neg);
    }

    void train() {

        CvANN_MLP bp;

        // Set up BPNetwork's parameters
        CvANN_MLP_TrainParams params;
        params.train_method = CvANN_MLP_TrainParams::BACKPROP;
        params.bp_dw_scale = 0.1;
        params.bp_moment_scale = 0.1;
        // params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,90000,0.00001);  //���ý�������
        params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001); //���ý�������
        //params.train_method=CvANN_MLP_TrainParams::RPROP;
        //params.rp_dw0 = 0.1;
        //params.rp_dw_plus = 1.2;
        //params.rp_dw_minus = 0.5;
        //params.rp_dw_min = FLT_EPSILON;
        //params.rp_dw_max = 50.;

        //Setup the BPNetwork
        // Mat layer_sizes=(cv::Mat_<int>(1,4) << feature.get_size(),int(feature.get_size()/2),int(feature.get_size()/2),class_num);

        cout << "input layer:" << feature.get_size() << " output layer:" << class_num << endl;

        cout << "layer_sizes:" << layer_sizes << endl;

        // Mat layer_sizes=(cv::Mat_<int>(1,3) << feature.get_size(),84,class_num);
        bp.create(layer_sizes, CvANN_MLP::SIGMOID_SYM, 1.0, 1.0);//CvANN_MLP::SIGMOID_SYM
        //CvANN_MLP::GAUSSIAN
        //CvANN_MLP::IDENTITY
        cout << "training...." << endl;

        cout << "sample size: pos+neg " << sample_feature_mat.rows << endl;

        bp.train(sample_feature_mat, sample_label_mat, Mat(), Mat(), params);


        bp.save(model_CNNs_path.c_str()); //save classifier
        cout << "training finish...Model_CNNs_Model.xml saved " << endl;

        bp.clear();
    }

    Mat get_sample_feature_mat() {
        return sample_feature_mat;
    }

    Mat get_sample_label_mat() {
        return sample_label_mat;
    }

    void set_model_version(string _version_CNNs) {
        version_CNNs = _version_CNNs;
        model_CNNs_path.assign("model_xml/Model_CNNs_").append(version_CNNs).append(".xml");
    }

    string get_model_CNNs_path() {
        return model_CNNs_path;
    }

    string get_model_version() {
        return version_CNNs;
    }

    void reset_layer(Mat _layer_sizes) {
        layer_sizes = _layer_sizes;
    }

    Mat get_layer() {
        return layer_sizes;
    }

//constructor
    pedestrian_detection_CNNs_trainer(pedestrian_detection_CNNs_feature _feature, string _version_CNNs) {
        feature = _feature;
        // layer_sizes=(cv::Mat_<int>(1,3) << feature.get_size(),500,class_num);
        layer_sizes = (cv::Mat_<int>(1, 4) << feature.get_size(), 300, 100, class_num);

        set_model_version(_version_CNNs);
    }


};

#endif //PEDESTRIAN_DETECTION_CNN_PEDESTRIAN_DETECTION_CNNS_TRAINER_HPP
