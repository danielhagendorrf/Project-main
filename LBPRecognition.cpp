
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "LBPRecognition.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

/*
	recieves : The function gets an image(Mat) , a cascade of faces
	           and a model of recognition.

	goal :     The function predicts if the face is the face of the
	           person in the database.

	returns :  The function return 0 if the man is recognised. Else
	           it returns 1.
*/

int LBP(Mat img, CascadeClassifier face_cascade, Ptr<FaceRecognizer> model)
{
	int test = 0;
	int predicted_label = -1;
	double predicted_confidence = 0.0;
	model->predict(img, predicted_label, predicted_confidence);
	string result_message;
	if (predicted_label == 0 && predicted_confidence > 5) {
		return 0;
	}
	else {
		return 1;
	}
}

Ptr<FaceRecognizer> trainLBP(vector<Mat>& images, vector<int>& labels)
{
	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer(2,16);
	model->train(images, labels);
	return model;
}


