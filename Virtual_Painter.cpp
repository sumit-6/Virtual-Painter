#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;


Mat img;
vector<vector<int>> newPoints;
Point getContours(Mat imgDil)
{
	// Getting Contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	Point myPoint(0, 0);
	findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Filtering out the variations
	double area, peri;
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());
	string objectType;
	for (int i = 0; i < contours.size(); i++)
	{
		area = (double)contourArea(contours[i]);
		//cout << area << "\n";
		if (area > 1000.0)
		{
			peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

			boundRect[i] = (boundingRect(conPoly[i]));
			myPoint.x = boundRect[i].x + boundRect[i].width / 2;
			myPoint.y = boundRect[i].y;
			drawContours(img, conPoly, i, Scalar(255, 0, 255), 1);

		}
	}
	return myPoint;
}

vector<vector<int>> myColors = { {70, 178, 88, 179, 255, 204} };
vector<Scalar> myColorValues = { { 0, 0, 255 } };

vector<vector<int>> findColor()
{
	Mat imgHSV;

	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	Mat mask;
	for (int i = 0; i < myColors.size(); i++)
	{
		Scalar lower(myColors[i][0], myColors[i][1], myColors[i][2]);
		Scalar upper(myColors[i][3], myColors[i][4], myColors[i][5]);
		inRange(imgHSV, lower, upper, mask);
		//imshow(to_string(i), mask);
		Point myPoint = getContours(mask);
		if (myPoint.x != 0 && myPoint.y != 0)
		{
			newPoints.push_back({ myPoint.x, myPoint.y, i });
		}
	}
	return newPoints;
}

void drawOnCanvas(vector<vector<int>> myPoints, vector<Scalar> color)
{
	for (int i = 0; i < myPoints.size(); i++)
	{
		circle(img, Point(myPoints[i][0], myPoints[i][1]), 10, color[myPoints[i][2]], FILLED);
	}
}

void VirtualPainter()
{
	VideoCapture cap(0);
	float w = 640, h = 480;

	while (true)
	{
		cap.read(img);
		vector<vector<int>> newPoints = findColor();
		drawOnCanvas(newPoints, myColorValues);

		Point2f source[4] = { {0.0f,0.0f}, {w,0.0f}, {0.0f, h}, {w,h} };
		Point2f destination[4] = { {w,0.0f}, {0.0f,0.0f}, {w,h}, {0.0f, h} };

		auto matrix = getPerspectiveTransform(source, destination);
		warpPerspective(img, img, matrix, Point(w, h));

		imshow("Image", img);
		waitKey(1);
	}
}

int main()
{
	VirtualPainter();
	return 0;
}
