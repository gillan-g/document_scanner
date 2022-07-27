#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

Mat img, nominal, imgResize, imgGray, imgBW, imOpen, imBlur, imCanny, imDialte, imgWarp;
int BW_threshold = 200;
int closeIterations = 3;
int canny_low_thresh = 100, canny_high_thresh = 200;
float w = 250, h = 350;


void Preview(Mat& img, string title)
{
	Mat disp;
	resize(img, disp, Size(), 0.33, 0.33);
	imshow(title, disp);

}

vector<Point> sortPoints(vector<Point> & points)
{
	vector<Point> newPoints;
	vector<int> sumPoints, subPoints;
	for (int i = 0; i < 4; i++)
	{
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);
	}
	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]);
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]);
	return newPoints;

}


Mat transformImage(Mat & nominal_img, vector<Point> & sortedPoints, float & w, float & h)
{
	Mat imgWarp, matrix;
	Point2f src[4] = { sortedPoints[0], sortedPoints[1], sortedPoints[2], sortedPoints[3] };
	Point2f dst[4] = { {0.0f, 0.0f}, {w, 0.0f}, {0.0f, h}, {w, h} };
	matrix = getPerspectiveTransform(src, dst);
	warpPerspective(nominal_img, imgWarp, matrix, Point(w, h));
	return imgWarp;
}

Mat applyCLAH(Mat& inImage)
{

	Mat lab_image;
	cvtColor(inImage, lab_image, COLOR_BGR2Lab);

	// Extract the L channel
	vector<Mat> lab_planes(3);
	split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

	// apply the CLAHE algorithm to the L channel
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(4);
	Mat dst;
	clahe->apply(lab_planes[0], dst);
	//clahe->apply(lab_planes[1], dst);
	//clahe->apply(lab_planes[2], dst);

	// Merge the the color planes back into an Lab image
	dst.copyTo(lab_planes[0]);
	//dst.copyTo(lab_planes[1]);
	//dst.copyTo(lab_planes[2]);
	merge(lab_planes, lab_image);

	// convert back to RGB
	Mat image_clahe;
	cvtColor(lab_image, image_clahe, COLOR_Lab2BGR);

	return image_clahe;
}
int main()
{
    VideoCapture cap(1);
	cap.set(CAP_PROP_FRAME_WIDTH, 1920/2); // valueX = your wanted width
	cap.set(CAP_PROP_FRAME_HEIGHT, 1080/2); // valueY = your wanted heigth

	namedWindow("Trackbars", (640, 200));
	createTrackbar("BW Threshold", "Trackbars", &BW_threshold, 255);
	//createTrackbar("Canny low", "Trackbars", &canny_low_thresh, 300);
	//createTrackbar("Canny high", "Trackbars", &canny_high_thresh, 300);
	createTrackbar("Open Iterations", "Trackbars", &closeIterations, 10);
	while (true) {
		cap.read(nominal);
		//resize(img, imgResize, Size(), 0.33, 0.33);

		// Convert to gray
		img = nominal.clone();
		cvtColor(img, imgGray, COLOR_RGB2GRAY);

		// Gaussian blur
		GaussianBlur(imgGray, imBlur, Size(5, 5), 1.0, 1.0);
		
		// Create BW image
		threshold(imBlur, imgBW, BW_threshold, 255, 0);
		Preview(imgBW, "BW Image");

		
		// Open to remove text
		int morph_size = 1;
		
		Mat element = getStructuringElement(
			MORPH_RECT,Size(2 * morph_size + 1,2 * morph_size + 1),
			Point(morph_size,morph_size));
		morphologyEx(imgBW, imOpen,MORPH_CLOSE, element, Point(-1, -1), closeIterations);
		Preview(imOpen, "Open Image");
		


		// Canny
		Canny(imOpen, imCanny, canny_low_thresh, canny_high_thresh);

		// Dilate
		dilate(imCanny, imDialte, element);

		// Extract largest countour
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		vector<Point> large_contour;
		vector<Point> conPoly, sorted_conPoly;

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
		
		// Display only if 4 corners are detected


		if (max_area > 0)
		{

			drawContours(img, vector<vector<Point> >(1, large_contour), -1, Scalar(0, 255, 0), 2);

			// Extract corners
			
			float peri = arcLength(large_contour, true);
			approxPolyDP(large_contour, conPoly, 0.02 * peri, true);

			if (conPoly.size() == 4)
			{
				sorted_conPoly = sortPoints(conPoly);

				// Test, add corners to image
				for (int i = 0; i < conPoly.size(); i++)
				{
					putText(img, to_string(i), Point(sorted_conPoly[i].x, sorted_conPoly[i].y),
						FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 0, 255), 2);
				}

				imgWarp = transformImage(nominal, sorted_conPoly, w, h);

				Mat clah_img;
				clah_img = applyCLAH(imgWarp);


 				imshow("Warped Image", clah_img);

			}
		};


		// Transform image
		Preview(img, "Image");;
		

		char key = (char)cv::waitKey(1);   // explicit cast
		if (key == 27) break;                // break if `esc' key was pressed. 
	}
}
