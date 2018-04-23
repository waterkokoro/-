// haisen.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include<opencv2\imgproc\imgproc.hpp>  
#include<opencv2\opencv.hpp>  
#include<opencv2\highgui\highgui.hpp>  
#include<vector>
using namespace cv;
using namespace std;
int main()
{
	Mat img0 = imread("234.jpg", 1);
	Mat img, img1, img2;
	img1 = img0.clone();
	img2 = img0.clone();
	cvtColor(img0, img0, CV_BGR2GRAY); //ת��ɫ�ʿռ�
	img = img0.clone();
	threshold(img, img, 128, 255, CV_THRESH_BINARY); //ͼ���ֵ�ڰ״���

/********************  ��˹�˲�  *********************************************************************/
	img.convertTo(img, CV_32FC1);
	GaussianBlur(img, img, Size(0, 0), 6, 6);

/********************  ƫ����  ***********************************************************************/
	//һ��ƫ����
	Mat m1, m2;
	m1 = (Mat_<float>(1, 2) << 1, -1);  //xƫ��
	m2 = (Mat_<float>(2, 1) << 1, -1);  //yƫ��

	Mat dx, dy;
	filter2D(img, dx, CV_32FC1, m1);
	filter2D(img, dy, CV_32FC1, m2);

	//����ƫ����
	Mat m3, m4, m5;
	m3 = (Mat_<float>(1, 3) << 1, -2, 1);   //����xƫ��
	m4 = (Mat_<float>(3, 1) << 1, -2, 1);   //����yƫ��
	m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);   //����xyƫ��

	Mat dxx, dyy, dxy;
	filter2D(img, dxx, CV_32FC1, m3);
	filter2D(img, dyy, CV_32FC1, m4);
	filter2D(img, dxy, CV_32FC1, m5);

/******************  hessian���� *************************************************************************/
	double maxD = -1;
	int imgcol = img.cols;
	int imgrow = img.rows;
	vector<double> Pt;
	for (int i = 0; i<imgcol; i++)
	{
		for (int j = 0; j<imgrow; j++)
		{
			if (img0.at<uchar>(j, i)>200)
			{
				Mat hessian(2, 2, CV_32FC1);
				hessian.at<float>(0, 0) = dxx.at<float>(j, i);
				hessian.at<float>(0, 1) = dxy.at<float>(j, i);
				hessian.at<float>(1, 0) = dxy.at<float>(j, i);
				hessian.at<float>(1, 1) = dyy.at<float>(j, i);

				Mat eValue;
				Mat eVectors;
				eigen(hessian, eValue, eVectors);

				double nx, ny;
				double fmaxD = 0;
				if (fabs(eValue.at<float>(0, 0)) >= fabs(eValue.at<float>(1, 0)))  //������ֵ���ʱ��Ӧ����������
				{
					nx = eVectors.at<float>(0, 0);
					ny = eVectors.at<float>(0, 1);
					fmaxD = eValue.at<float>(0, 0);
				}
				else
				{
					nx = eVectors.at<float>(1, 0);
					ny = eVectors.at<float>(1, 1);
					fmaxD = eValue.at<float>(1, 0);
				}

				double t = -(nx*dx.at<float>(j, i) + ny*dy.at<float>(j, i)) / (nx*nx*dxx.at<float>(j, i) + 2 * nx*ny*dxy.at<float>(j, i) + ny*ny*dyy.at<float>(j, i));

				if (fabs(t*nx) <= 0.5 && fabs(t*ny) <= 0.5)
				{
					Pt.push_back(i);
					Pt.push_back(j);
				}
			}
		}
	}
/******************* �������������ߵĵ㣬�ŵ�up��down������ **********************************************/
	int Maxx, Maxy = -1; //�ú��������ͼ�����ĵ���Ϊɸѡ���µ�ķֽ���
	for (int k = 0; k<Pt.size() / 2; k++)
	{
		Point rpt;
		rpt.x = Pt[2 * k + 0];
		rpt.y = Pt[2 * k + 1];
		if (rpt.y >= Maxy)
		{
			Maxy = rpt.y;
			Maxx = rpt.x;
		}
		cout << rpt.x << "," << rpt.y << endl; //�������
	}
	cout << Maxx << ",,,,,�ֽ��,,,,," << Maxy << endl;
	vector<Point> up;
	vector<Point> down;
	for (int i = 0; i < Pt.size() / 2; i++)
	{
		if (Pt[2 * i + 1] < Maxx)
		{
			down.push_back(Point(Pt[2 * i + 0], Pt[2 * i + 1]));
		}
		else
		{
			up.push_back(Point(Pt[2 * i + 0], Pt[2 * i + 1]));
		}
	}

	for (int k = 0; k<Pt.size() / 2; k++)
	{
		Point rpt;
		rpt.x = Pt[2 * k + 0];
		rpt.y = Pt[2 * k + 1];
		circle(img1, rpt, 5, Scalar(0, 0, 255));
	}
/*********************  ���ֱ��  ****************************************************************/
	Vec4f line_1;
	fitLine(up, line_1, DIST_L2, 0, 1e-2, 1e-2); //��Ϻ���
	std::cout << "line_1 = " << line_1 << std::endl;

	//��ȡline1��бʽ�ĵ��б��  
	Point point0;
	point0.x = line_1[2];
	point0.y = line_1[3];
	double k = line_1[1] / line_1[0];

	//����ֱ�ߵĶ˵�(y = k(x - x0) + y0)  
	Point point1, point2;
	point1.x = 0;
	point1.y = k * (0 - point0.x) + point0.y;
	point2.x = 640;
	point2.y = k * (640 - point0.x) + point0.y;
	double b1 = point1.y - k *  point1.x;

	//line(img1, point1, point2, Scalar(0, 255, 0), 2, 8, 0);

	Vec4f line_2;
	fitLine(down, line_2, DIST_L2, 0, 1e-2, 1e-2); //��Ϻ���

	std::cout << "line_2 = " << line_2 << std::endl;

	//��ȡline2��бʽ�ĵ��б��  
	Point point3;
	point3.x = line_2[2];
	point3.y = line_2[3];
	double k1 = line_2[1] / line_2[0];

	//����ֱ�ߵĶ˵�(y = k(x - x0) + y0)  
	Point point4, point5;
	point4.x = 0;
	point4.y = k1 * (0 - point3.x) + point3.y;
	point5.x = 640;
	point5.y = k1 * (640 - point3.x) + point3.y;
	double b2 = point4.y - k *  point4.x;

	//line(img1, point4, point5, Scalar(0, 255, 0), 2, 8, 0);

/*****************************  ���㽻��  **********************************************/
	double centerx = (b2 - b1) / (k - k1);
	double centery = k * centerx + b1;

	cout << "����X: " << centerx << "����Y: " << centery << endl;
	circle(img2, Point(centerx, centery), 10, Scalar(0, 255, 255));
	line(img2, Point(centerx, centery), point2, Scalar(0, 255, 0), 2, 8, 0);
	line(img2, Point(centerx, centery), point5, Scalar(0, 255, 0), 2, 8, 0);

	imshow("����ͼ", img1);
	imshow("���ս��", img2);

	imwrite("�����������ص�ͼ.jpg", img1);
	imwrite("���ս��--�������Լ����߽���.jpg", img2);

	waitKey(0);
	return 0;
}

