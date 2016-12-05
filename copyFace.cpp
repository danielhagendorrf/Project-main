#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "copyFace.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

Mat copyFace(Mat img,int leftWidth,int bottomHeight,int rightWidth,int topHeight) {
	Mat copy;
	Mat image;
	//image.create(Size(rightWidth - 50 - leftWidth - 25, topHeight - bottomHeight), img.type());
	copy.create(img.size(), img.type());
	copy.setTo(Scalar(0,0,0));
	//int x, y;
	//x = 0;
	for (int i = bottomHeight; i < topHeight; i++) {
		//y = 0;
		for (int j = leftWidth; j < rightWidth; j++) {
			//image.at<Vec3b>(x, y) = img.at<Vec3b>(i, j);
			copy.at<Vec3b>(i-bottomHeight, j-leftWidth) = img.at<Vec3b>(i, j);
			//copy.at<8uc1>(i - bottomHeight, j - leftWidth) = img.at<8uc1>(i, j);
			//y++;
		}
		//x++;
	}
	//equalizeHist(image, image);
	//namedWindow("copy", WINDOW_AUTOSIZE);
	//imshow("copy", image);
	return copy;
}