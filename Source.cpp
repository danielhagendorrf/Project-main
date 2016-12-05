#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "C:/Users/Daniel Hagendorf/Documents/Visual Studio 2015/Projects/Project6/Project6/eigenfaceRecognition.h"
#include "C:/Users/Daniel Hagendorf/Documents/Visual Studio 2015/Projects/Project6/Project6/LBPRecognition.h"
#include "copyFace.h"
#include "readCSV.h"
#include "fisherface.h"
#include "alignFace.h"
#include "colorBalancing.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

using namespace std;
using namespace cv;


void detectAndDisplay(Mat frame, vector<Mat>& images, vector<int>& labels);
int eigen(Mat img, CascadeClassifier face_cascade, Ptr<BasicFaceRecognizer> model);
int LBP(Mat img, CascadeClassifier face_cascade, Ptr<FaceRecognizer> model);
Ptr<FaceRecognizer> trainLBP(vector<Mat>& images, vector<int>& labels);
Ptr<BasicFaceRecognizer> train(vector<Mat>& images, vector<int>& labels);
int fisher(Mat img, CascadeClassifier face_cascade, vector<Mat>& images, vector<int>& labels);
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator, CascadeClassifier face_cascade);
Mat copyFace(Mat img, int leftWidth, int bottomHeight, int rightWidth, int topHeight);
void alignFaceWithEyes(Mat img, int lEyeY, int rEyeY, int imgW, int imgH);
void colorBalancing(Mat& img, Mat& rImg, float percent);

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";


int main(void)
{
	Mat img;
	img = imread("C:/Users/Daniel Hagendorf/Pictures/Camera Roll/p8.jpg");
	//Size size(1280,720);
	//if (img.size() != size)
		//resize(img, img, size);
	//cout << size.height << endl;
	//cout << size.width << endl;
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };

	string csv = string("c:/csv2.csv");
	vector<Mat> images;
	vector<int> labels;
	try {
		read_csv(csv, images, labels, ';', face_cascade);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(Error::StsError, error_message);
	}
	detectAndDisplay(img,images,labels);
	waitKey(0);
	return 0;
}

void detectAndDisplay(Mat frame, vector<Mat>& images, vector<int>& labels)
{
	Mat frame1;
	colorBalancing(frame, frame1, 20.0f);
	namedWindow("hhh", WINDOW_AUTOSIZE);
	imshow("hhh", frame1);
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame1, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	int prediction=0;
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame1, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		Mat face = copyFace(frame1,faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		cvtColor(face, face,COLOR_BGR2GRAY);
		namedWindow("Daniel", WINDOW_AUTOSIZE);
		imshow("Daniel", face);
		Ptr<BasicFaceRecognizer> model = train(images, labels);
		prediction=eigen(face, face_cascade,model);
		if (prediction = 0) {
			cout << "the man is in the database" << endl;
		}
		else {
			cout << "not recognised" << endl;
		}
		Ptr<FaceRecognizer> model2 = trainLBP(images, labels);
		prediction = LBP(face, face_cascade, model);
		if (prediction = 0) {
			cout << "the man is in the database" << endl;
		}
		else {
			cout << "not recognised" << endl;
		}
	}
	imshow(window_name, frame);
}

