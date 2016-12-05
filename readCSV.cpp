#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "copyFace.h"
#include "colorBalancing.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>

using namespace std;
using namespace cv;


void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator, CascadeClassifier face_cascade) {
	//Size size(1280, 720);
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		int oneTime = 0;
		if (!path.empty() && !classlabel.empty()) {
			std::vector<Rect> faces;
			Mat img = imread(path);
			//if (img.size() != size)
			//resize(img, img, size);
			Mat rImg;
			if (!img.empty()) {
				//cvtColor(img, img, COLOR_BGR2GRAY);
				//equalizeHist(img, img);
				face_cascade.detectMultiScale(img, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
				if (faces.size() == 1) {
					Mat face = copyFace(img, faces[0].x, faces[0].y, faces[0].x + faces[0].width, faces[0].y + faces[0].height);
					cvtColor(face, face, COLOR_BGR2GRAY);
					images.push_back(face);
					labels.push_back(atoi(classlabel.c_str()));
				}
				else {
					for (float i = 1; i < 11; i++) {
						colorBalancing(img, rImg, i);
						face_cascade.detectMultiScale(rImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
						if (faces.size() == 1) {
							Mat face = copyFace(rImg, faces[0].x, faces[0].y, faces[0].x + faces[0].width, faces[0].y + faces[0].height);
							cvtColor(face, face, COLOR_BGR2GRAY);
							images.push_back(face);
							labels.push_back(atoi(classlabel.c_str()));
							i = 11;
						}
						else {
							oneTime++;
							if (oneTime == 10) {
								cout << "could not load image" << endl;
								cout << path << endl;
								oneTime = 1;
							}
						}
					}
				}


			}
		}
	}
}