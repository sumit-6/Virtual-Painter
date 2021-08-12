#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

void ImageProcessing()
{
	string path = "PendulumDifferentialEquation.png";
	Mat img = imread(path);
	resize(img, img, Size(1000, 500));
	imshow("Image", img);
	waitKey(0);
}

void PlayVideo(string path)
{
	VideoCapture cap(path);
	Mat IMG;
	while (true)
	{
		cap.read(IMG);
		imshow("Video Player",IMG);
		waitKey(20);
	}
}

void WebCam()
{
	VideoCapture cap(0);
	Mat Image_Data;
	while (true)
	{
		cap.read(Image_Data);
		imshow("Web Camera", Image_Data);
		waitKey(10);
	}
}

void imageProcess()
{
	string path = "PendulumDifferentialEquation.png";
	Mat img = imread(path);
	Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

	resize(img, img, Size(1000, 500));

	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);
	erode(imgDil, imgErode, kernel);

	imshow("Image", img);
	imshow("Image Gray", imgGray);
	imshow("Image Blur", imgBlur);
	imshow("Image Canny", imgCanny);
	imshow("Image Dilation", imgDil);
	imshow("Image Erode", imgErode);
	waitKey(0);
}

void IMGResize()
{
	string path = "PendulumDifferentialEquation.png";
	Mat img = imread(path);
	Mat imgResized;
	//cout << img.size() << "\n"; // [1920 x 1080]
	//resize(img, imgResized, Size(1000,500));
	resize(img, imgResized, Size(), 0.5, 0.5);
	imshow("Image", img);
	imshow("Image Resized", imgResized);
	waitKey(0);
}

void IMGCrop()
{
	string path = "PendulumDifferentialEquation.png";
	Mat img = imread(path);
	resize(img, img, Size(1000, 500));
	Mat imgCropped;

	Rect roi(100, 0, 300, 250);
	imgCropped = img(roi);
	imshow("Image", img);
	imshow("Image Cropped", imgCropped);
	waitKey(0);
}

void DrawShapes()
{
	//Blank Image
	auto BackGroundColour = Scalar(255, 255, 255);
	Mat img(512, 512, CV_8UC3, BackGroundColour);
	auto centre = Point(256, 256);
	int radius = 156;
	auto colour = Scalar(0, 69, 255);
	//circle(img, centre, radius, colour);
	circle(img, centre, radius, colour, FILLED);

	colour = Scalar(255, 255, 255);
	rectangle(img, Point(130, 226), Point(382, 286), colour, FILLED);

	line(img, Point(130, 296), Point(382, 296), colour, 2);

	putText(img, "Sumit Verma", Point(137, 262), FONT_HERSHEY_TRIPLEX, 1.05, Scalar(0, 69, 255), 2);

	imshow("Image", img);
	waitKey(0);
}

void IMGwarp()
{
	string path = "cards.jpg";
	Mat img = imread(path);
	float w = 350, h = 200;
	Mat matrix, imgWarp;

	Point2f source[4] = { {313, 588}, {1135, 539}, {64, 940} ,{924, 949} };
	Point2f destination[4] = { {0.0f,0.0f}, {w,0.0f}, {0.0f, h}, {w,h} };

	matrix = getPerspectiveTransform(source, destination);
	warpPerspective(img, imgWarp, matrix, Point(w, h));

	//imshow("Image", img);
	imshow("Warped Image", imgWarp);

	waitKey(0);
} 

void colorDetection()
{
	string path = "PendulumDifferentialEquation.png";
	Mat img = imread(path);
	Mat imgHSV, mask;
	resize(img, img, Size(1000, 500));
	
	int hmin = 0, smin = 0, vmin = 0;
	int hmax = 179, smax = 255, vmax = 255;
	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	
	namedWindow("TrackBar", (640, 200));
	createTrackbar("Hue Min", "TrackBar", &hmin, 179);
	createTrackbar("Hue Max", "TrackBar", &hmax, 179);
	createTrackbar("Saturation Min", "TrackBar", &smin, 255);
	createTrackbar("Saturation Max", "TrackBar", &smax, 255);
	createTrackbar("Value Min", "TrackBar", &vmin, 255);
	createTrackbar("Value Max", "TrackBar", &vmax, 255);

	while (true)
	{
		Scalar lower(hmin, smin, vmin);
		Scalar upper(hmax, smax, vmax);
		inRange(imgHSV, lower, upper, mask);

		imshow("Image", img);
		//imshow("Image HSV", imgHSV);
		imshow("Image mask", mask);
		waitKey(1);

	}
}

void shapeDetection()
{
	string path = "shapes.png";
	Mat img = imread(path);
	Mat imgGray, imgBlur, imgCanny, imgDil;

	// Preprocessing
	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);

	// Getting Contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
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
			approxPolyDP(contours[i], conPoly[i], 0.02*peri, true);
			
			boundRect[i] = (boundingRect(conPoly[i]));
			
			int objCorner = (int)conPoly[i].size();
			if (objCorner == 4)
			{
				double aspRatio = (double)boundRect[i].width / (double)boundRect[i].height;
				if (aspRatio >= 0.95 && aspRatio <= 1.05)
				{
					objectType = "Square";
				}
				else
				objectType = "Rect";
			}
			else if (objCorner == 3)
			{
				objectType = "Tri";
			}
			else if (objCorner > 4)
			{
				objectType = "Circle";
			}
			drawContours(img, conPoly, i, Scalar(255, 0, 255), 1);
			//cout << conPoly[i].size() << "\n";
			rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 4);
			putText(img, objectType, { boundRect[i].x, boundRect[i].y - 5 }, FONT_ITALIC, 0.60, Scalar(0, 69, 255), 1);
		}
	}

	// Printing the image
	imshow("Image", img);
	/*imshow("Image Gray", imgGray);
	imshow("Image Blur", imgBlur);
	imshow("Image Canyy", imgCanny);
	imshow("Image Dil", imgDil);*/
	waitKey(0);
}

void faceRecognition()
{
	VideoCapture cap(0);
	Mat Image_Data;
	vector<Rect> faces;
	CascadeClassifier faceCascade;
	faceCascade.load("C:/Users/verma/Downloads/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml");
	while (true)
	{
		cap.read(Image_Data);
		
		int x = 1;
		faceCascade.detectMultiScale(Image_Data, faces, 1.1, 10);
		try
		{
			for (int i = 0; i < faces.size(); i++)
			{
				rectangle(Image_Data, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 2);
			}
			throw x;
		}
		catch (int x)
		{
			imshow("Web Camera", Image_Data);
		}
		catch (...)
		{
			imshow("Web Camera", Image_Data);
		}
		
		waitKey(10);
	}
}

void findColor(Mat img)
{
	Mat imgHSV;
	vector<vector<int>> myColors = { {},
									 {},
									 {} };
	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	Scalar lower(hmin, smin, vmin);
	Scalar upper(hmax, smax, vmax);
	inRange(imgHSV, lower, upper, mask);
}

void VirtualPainter()
{
	VideoCapture cap(0);
	Mat img;
	while (true)
	{
		cap.read(img);
		findColor(img);
		imshow("Image", img);
		waitKey(1);
	}
}

int main()
{
	//ImageProcessing();
	//WebCam();
	//PlayVideo("Double Pendulum Project.mp4");
//------------------------------------------------------------
	//  **functions**
	//	imageProcess();
//------------------------------------------------------------
// 
	//IMGResize();
	//IMGCrop();
	//DrawShapes();
	//IMGwarp();
	//colorDetection();
	//shapeDetection();
	//faceRecognition();
	VirtualPainter();
	return 0;
}