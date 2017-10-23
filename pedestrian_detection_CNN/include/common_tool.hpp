//
// Created by 翁规范 on 10/22/17.
//

#ifndef PEDESTRIAN_DETECTION_CNN_COMMON_TOOL_HPP
#define PEDESTRIAN_DETECTION_CNN_COMMON_TOOL_HPP


#include <time.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
// #include <hog2_without_IPP_6.hpp>

// using cv::Mat;
// using cv::Point;
// using cv::HOGDescriptor;
// using cv::Size;
// using cv::imshow;
// using cv::imwrite;
// using cv::imread;
// using cv::Scalar;
// using cv::Rect;

using namespace std;

string get_file_index(string file_nam_integral) {

    int sub_index0 = file_nam_integral.rfind("/"), sub_index1 = file_nam_integral.rfind(".");

    return file_nam_integral.substr(sub_index0 + 1, sub_index1 - sub_index0 - 1);

}


void get_files(string path, vector<string> &files) {

    DIR *dir;
    struct dirent *ptr;
    dir = opendir(path.c_str());

    string pathName, exdName;

    while ((ptr = readdir(dir)) != NULL) {

        if (ptr->d_name[0] != '.' && ptr->d_name[strlen(ptr->d_name) - 4] == '.') {
            files.push_back(pathName.assign(path).append("\\").append(string(ptr->d_name)));
            // files.push_back(string(ptr->d_name));

        }

    }

    // for(i=0;i<files.size();++i){
    // cout<<files[i]<<endl;
    // }
}

void get_files_last_name(string path, vector<string> &files) {

    DIR *dir;
    struct dirent *ptr;
    dir = opendir(path.c_str());

    string pathName, exdName;

    while ((ptr = readdir(dir)) != NULL) {

        if (ptr->d_name[0] != '.' && ptr->d_name[strlen(ptr->d_name) - 4] == '.') {
            // files.push_back(pathName.assign(path).append("\\").append(string(ptr->d_name)));
            files.push_back(string(ptr->d_name));

        }

    }

    // for(i=0;i<files.size();++i){
    // cout<<files[i]<<endl;
    // }
}

string get_time_stamp() {

    time_t t = time(0);
    char time_str[100];

    strftime(time_str, sizeof(time_str), "%Y%m%d_%H%M%S", localtime(&t));

    return string(time_str);
}


void copy_file(string src, string dst) {

    ifstream in;

    in.open(src, ios_base::binary);

    if (!in) {
        std::cout << "open src file : " << src << " failed" << std::endl;
        return;
    }

    std::ofstream out;
    out.open(dst, ios_base::binary);
    if (!out) {
        std::cout << "create new file : " << dst << " failed" << std::endl;
        in.close();
        return;
    }

    out << in.rdbuf();

    // if(!in||!out){cerr<<"Open File Failure,Please Try Again!";exit(1);}
    // while(!in.eof())
    // {
    // in.read(buffer,256);       //���ļ��ж�ȡ256���ֽڵ����ݵ�������
    // n=in.gcount();             //��������һ�в�֪��ȡ�˶����ֽڵ����ݣ������ú�������һ�¡�
    // out.write(buffer,n);       //д���Ǹ��ֽڵ�����
    // }

    out.close();
    in.close();
}


#endif //PEDESTRIAN_DETECTION_CNN_COMMON_TOOL_HPP
