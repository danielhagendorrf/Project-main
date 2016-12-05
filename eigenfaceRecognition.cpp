
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "C:/Users/Daniel Hagendorf/Documents/Visual Studio 2015/Projects/Project6/Project6/eigenfaceRecognition.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;


int eigen(Mat img, CascadeClassifier face_cascade, Ptr<BasicFaceRecognizer> model) {

	int test = 0;
	int predicted_label = -1;
	double predicted_confidence = 0.0;
	model->predict(img, predicted_label, predicted_confidence);
	string result_message;
	if (predicted_label == 0 && predicted_confidence > 10000) {
		return 0;
	}
	else {
		return 1;
	}
}

Ptr<BasicFaceRecognizer> train(vector<Mat>& images, vector<int>& labels) {
	Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer(0);
	model->train(images, labels);
	return model;
}


