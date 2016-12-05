#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "copyFace.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

using namespace std;
using namespace cv;


void colorBalancing(Mat& img, Mat& rImg, float percent) {
	assert(img.channels() == 3);
	assert(percent > 0 && percent < 100);

	float half_percent = percent / 200.0f;

	vector<Mat> tmpsplit; split(img, tmpsplit);
	for (int i = 0; i<3; i++) {
		Mat flat; tmpsplit[i].reshape(1, 1).copyTo(flat);
		cv::sort(flat, flat, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
		
		tmpsplit[i].setTo(lowval, tmpsplit[i] < lowval);
		tmpsplit[i].setTo(highval, tmpsplit[i] > highval);

		//scale the channel
		normalize(tmpsplit[i], tmpsplit[i], 0, 255, NORM_MINMAX);
	}
	merge(tmpsplit, rImg);
}