#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "eigenfaceRecognition.h"
#include "fisherfaceRecognition.h"
#include "LBPRecognition.h"
#include "copyFace.h"
#include "readCSV.h"
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
int fisher(Mat img, CascadeClassifier face_cascade, Ptr<BasicFaceRecognizer> model);
Ptr<BasicFaceRecognizer> trainF(vector<Mat>& images, vector<int>& labels);
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator, CascadeClassifier face_cascade);
Mat copyFace(Mat img, int leftWidth, int bottomHeight, int rightWidth, int topHeight);
void colorBalancing(Mat& img, Mat& rImg, float percent);
Mat pictureToAnalise(vector<Mat> images);
void predictFace(vector<Rect> faces, Mat frame, Mat frame_gray, Ptr<BasicFaceRecognizer> model);

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";


/*
	recieves : The function recieves nothing.

	goal :	   The function checks if the files given in the begining of the program
		       corespond with the requierments of the project. Plus, the function
		       initiates the function "detectAndDisplay".

	returne :  The function return -1 if it exited because of a problem with the 
	           cascades file , 1 if there is a problem with reading the csv file
			   and 0 if the code ran through.
*/

int main(void)
{
	Mat img;
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };
	/*string test = string("c:/Users/Daniel Hagendorf/test.csv");
	ofstream file(test, ios::out | ios::app);
	string s1 = "c:/WeizmanProjectPictures/Daniel/p2.jpg;0";
	string s2 = "c:/WeizmanProjectPictures/Daniel/p3.jpg;0";
	file << s1 << endl;
	file << s2 <<endl;
	file.close();*/
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
	img = pictureToAnalise(images);
	detectAndDisplay(img,images,labels);
	waitKey(0);
	return 0;
}

/*
	recieves : The function recieves a vector of images.

	goal :     The function opens the camera and when the
	           user is ready takes a picture of him. Plus,
			   it crops the frame so the picture will have
			   the same scale as the training images;

	returns :  The function returns the image it captured.
*/

Mat pictureToAnalise(vector<Mat> images)
{
	Mat img;
	VideoCapture video;
	cout << "press the enter key when you are ready for your photo to be taken" << endl;
	getchar();
	if (!video.open(0)) {
		cout << "camera not working" << endl;
	}
	video.set(CAP_PROP_FRAME_HEIGHT, 720);
	video.set(CAP_PROP_FRAME_WIDTH, 1280);
	video.retrieve(img, images[0].type());
	video.release();
	return img;
}

/*
	recieves : The function recieves a vector of rectangles, two images
	           and a model of face recognition.

	goal :     The goal is to check for each face in the image if it is
	           recognised and to act accordingly.

	returns :  The function displays the face of the person and whether
			   he was recognised or not.
*/

void predictFace(vector<Rect> faces , Mat frame , Mat frame_gray , Ptr<BasicFaceRecognizer> model)
{
	int prediction = 0;
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);
		Mat face = copyFace(frame, faces[i].x, faces[i].y, faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		cvtColor(face, face, COLOR_BGR2GRAY);
		prediction = fisher(face, face_cascade, model);
		if (prediction == 0) {
			namedWindow("correct person", WINDOW_AUTOSIZE);
			imshow("correct person", face);
		}
		else {
			namedWindow("you are not recognised please try again", WINDOW_AUTOSIZE);
			imshow("you are not recognised please try again", face);
		}
		/*Ptr<FaceRecognizer> model2 = trainLBP(images, labels);
		prediction = LBP(face, face_cascade, model);
		if (prediction = 0) {
		cout << "the man is in the database" << endl;
		}
		else {
		cout << "not recognised" << endl;
		}*/
	}
}

/*
	recieves : The function recieves an image, a vector of images 
	           and a vector of integers.

	goal :     The function checks if there is faces in the image
	           and finds them. Plus, it prepares all the variables 
			   the function "predictFace" needs and calls it"

	returns :  The function displays the image the program took.
*/

void detectAndDisplay(Mat frame, vector<Mat>& images, vector<int>& labels)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame ,frame_gray ,COLOR_BGR2GRAY);
	equalizeHist(frame_gray ,frame_gray);
	Ptr<BasicFaceRecognizer> model = trainF(images, labels);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	if (faces.size() == 0)
		cout << "could not find a face in the picture" << endl;
	else
		predictFace(faces ,frame ,frame_gray ,model );
	imshow(window_name, frame);
}

