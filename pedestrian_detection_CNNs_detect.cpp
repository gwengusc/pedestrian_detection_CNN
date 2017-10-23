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
#include <sys/time.h>

#include <pedestrian_detection_CNNs_feature.hpp>
#include <pedestrian_detection_CNNs_detector.hpp>
// #include <pedestrian_detection_CNNs_trainer_170525_0.hpp>

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
using cv::namedWindow;
using cv::setMouseCallback;
using cv::waitKey;
using cv::saturate_cast;

using std::cin;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;


int main() {

        pedestrian_detection_CNNs_feature::init_max_map();

        struct timeval tpstart, tpend;
        double timeuse;

        pedestrian_detection_CNNs_feature feature(Size(32, 64), 3);

        pedestrian_detection_CNNs_detector detector(feature.get_size_max_pool());

        string src_name("/Users/wengguifan/pedestrian_data_test/pedestrian_image/INRIA/crop_000025.png");

        Mat src_img = imread(src_name, CV_LOAD_IMAGE_GRAYSCALE);

        Mat src_img_rgb_crop = imread(src_name);
        Mat src_img_rgb_show;

        src_img_rgb_crop.copyTo(src_img_rgb_show);

        gettimeofday(&tpstart, NULL);

        vector<Rect> targets = detector.detect_multi(src_img, 2, 0.85,
                                                     "/Users/wengguifan/CLionProjects/pedestrian_detection_CNN/model/Model_CNNs_20170727_115303_init_pos_400_neg_500_layer_210_80_2_train_1876_1994.xml");

        gettimeofday(&tpend, NULL);

        timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;

        timeuse /= 1000;

        cout << "timeuse:" << timeuse << "ms" << endl;


        cout << targets.size() << endl;


        int i, target_size = targets.size();

        Rect rect;

//    Mat sub_img;

        stringstream trans;

        for (i = 0; i < target_size; i++) {

                rect = targets[i];

//        cout<<"rect.x:"<<rect.x<<" rect.y:"<<rect.y<<" width:"<<rect.width<<" height:"<<rect.height<<endl;
                rectangle(src_img_rgb_show, rect.tl(), rect.br(), Scalar(0, 255, 0, 0), 1, 8,0);

//        sub_img = src_img_rgb_crop(rect);

//        trans << "D:\\pedestrian_detection_project\\pedestrian_detect_CNNs\\dst_img\\sub\\sub_img" << i << ".bmp";

//        imwrite(trans.str(), sub_img);

                trans.clear();
                trans.str("");

//        sub_img.release();
        }

        imwrite("D:\\pedestrian_detection_project\\pedestrian_detect_CNNs\\dst_img\\source_img_rect\\Pedestrian-Safety.jpg",
                src_img_rgb_show);

        imshow("src_img_rgb", src_img_rgb_show);


        waitKey(0);

        return 0;
}
