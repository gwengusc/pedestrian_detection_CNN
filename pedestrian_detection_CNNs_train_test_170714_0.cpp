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
#include <pedestrian_detection_CNNs_feature_170525_3.hpp>
//#include <pedestrian_detection_CNNs_feature_170627_2.hpp>
#include <pedestrian_detection_CNNs_trainer_170525_0.hpp>
#include <sys/time.h>

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

string sample_path_train_pos,sample_path_train_neg,version;

void config(){

	pedestrian_detection_CNNs_feature::init_max_map();

    sample_path_train_pos.assign("D:\\pedestrian_samples\\train_gray\\pos_gray\\main_400\\");
    sample_path_train_neg.assign("D:\\pedestrian_samples\\train_gray\\neg_gray\\main_500\\");
    version.assign(get_time_stamp()).append("_init_pos_400_neg_500_layer_465_200_2");
//    log_path.assign("log\\predict_rates_").append(version).append("_1000_1250.txt");
}

int main(){

	config();

	pedestrian_detection_CNNs_feature feature(Size(32,64),3);

	vector<string>sample_files_pos,sample_files_neg;

	get_files(sample_path_train_pos,sample_files_pos);
	get_files(sample_path_train_neg,sample_files_neg);

	
	int sample_neg_size=sample_files_neg.size(), sample_pos_size=sample_files_pos.size();
	
	cout<<"neg sample size:"<<sample_files_neg.size()<<endl;
	cout<<"pos sample size:"<<sample_files_pos.size()<<endl;
	
	stringstream trans;
	
	trans<<version<<"_train_"<<sample_pos_size<<"_"<<sample_neg_size;

	pedestrian_detection_CNNs_trainer trainer(feature,trans.str());

    trainer.reset_layer((cv::Mat_<int>(1,3) << feature.get_size(),200,2));

    struct timeval tpstart,tpend;
    double timeuse;

	trainer.construct_train_data(sample_files_pos,sample_files_neg);

    gettimeofday(&tpstart,NULL);

	trainer.train();

    gettimeofday(&tpend,NULL);

    timeuse=1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec;

    timeuse/=1000;

    cout<<"time_used: "<<timeuse<<"ms"<<endl;

	return 0;
}
