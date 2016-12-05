#pragma once
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace cv::face;
using namespace std;

Mat copyFace(Mat img, int leftWidth, int bottomHeight, int rightWidth, int topHeight);