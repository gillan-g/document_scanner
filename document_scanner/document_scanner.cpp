#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

Mat img, imgResize, imgGray, imgBW, imOpen, imBlur, imCanny, imDialte;
int BW_threshold = 170;
int canny_low_thresh = 100, canny_high_thresh = 200;

int main()
{
    VideoCapture cap(1);

	namedWindow("Trackbars", (640, 200));
	createTrackbar("BW Threshold", "Trackbars", &BW_threshold, 255);
	createTrackbar("Canny low", "Trackbars", &canny_low_thresh, 300);
	createTrackbar("Canny high", "Trackbars", &canny_high_thresh, 300);

	while (true) {
		cap.read(img);
		//resize(img, imgResize, Size(), 0.33, 0.33);

		// Convert to gray
		cvtColor(img, imgGray, COLOR_RGB2GRAY);
		
		// Create BW image
		threshold(imgGray, imgBW, BW_threshold, 255, 0);
		
		// Open to remove text
		int morph_size = 1;
		Mat element = getStructuringElement(
			MORPH_RECT,Size(2 * morph_size + 1,2 * morph_size + 1),
			Point(morph_size,morph_size));
		morphologyEx(imgBW, imOpen,MORPH_OPEN, element, Point(-1, -1), 1);
		
		// Gaussian blur
		GaussianBlur(imOpen, imBlur, Size(5, 5), 3.0, 3.0);

		// Canny
		Canny(imBlur, imCanny, canny_low_thresh, canny_high_thresh);

		// Dilate
		dilate(imCanny, imDialte, element);

		// Extract largest countour
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		vector<Point> large_contour;

		findContours(imDialte, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
		
		// find max area contours
		int max_area = 0;
		for (unsigned int i = 0; i < contours.size(); ++i) {
			int area = (int)contourArea(contours[i]);
			if (area > max_area) {
				large_contour = contours[i];
				max_area = area;
			}
		};
		
		//drawContours(img, vector<vector<Point> >(1, large_contour), -1, Scalar(0, 255, 0), 2);


		// Extract corners
		vector<Point> conPoly;
		float peri = arcLength(large_contour, true);
		approxPolyDP(large_contour, conPoly, 0.02 * peri, true);
		

		// Test, add corners to image
		for (int i = 0; i < conPoly.size(); i++) 
		{
			putText(img,to_string(i), Point(conPoly[i].x, conPoly[i].y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 2);
		}
		

		// Transform image
		imshow("Image", img);

		char key = (char)cv::waitKey(1);   // explicit cast
		if (key == 27) break;                // break if `esc' key was pressed. 
	}
}
