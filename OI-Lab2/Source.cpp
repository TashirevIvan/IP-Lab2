#include <iostream> 
#include <string>   
#include <iomanip>  
#include <sstream>  
#include <clocale>
#include <vector>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

template <class T>
T clamp(T v, int max, int min) {
	if (v > max)
		return max;

	else if (v < min)
		return min;

	return v;
}

Mat AddGaussianNoise(const Mat mSrc, double Mean = 0.0, double StdDev = 30.0) {
	Mat mDst(mSrc.size(), mSrc.type());

	Mat mGaussian_noise = Mat(mSrc.size(), CV_16SC3);
	randn(mGaussian_noise, Scalar::all(Mean), Scalar::all(StdDev));

	for (int Rows = 0; Rows < mSrc.rows; Rows++) {
		for (int Cols = 0; Cols < mSrc.cols; Cols++) {
			Vec3b Source_Pixel = mSrc.at<Vec3b>(Rows, Cols);
			Vec3b& Des_Pixel = mDst.at<Vec3b>(Rows, Cols);
			Vec3s Noise_Pixel = mGaussian_noise.at<Vec3s>(Rows, Cols);

			for (int i = 0; i < 3; i++) {
				int Dest_Pixel = Source_Pixel.val[i] + Noise_Pixel.val[i];
				Des_Pixel.val[i] = clamp<int>(Dest_Pixel, 255, 0);
			}
		}
	}

	return mDst;
}

float calculateNewPC(Mat photo, int x, int y, int rgb, int radius, int sigma) {
	float returnPC = 0;
	int size = 2 * radius + 1;
	float* vector = new float[size * size];
	float norm = 0;
	for (int i = -radius; i <= radius; i++) {
		for (int j = -radius; j <= radius; j++) {
			int idx = (i + radius) * size + j + radius;
			vector[idx] = exp(-(i * i + j * j) / sigma * sigma);
			norm += vector[idx];
		}
	}
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			vector[i * size + j] /= norm;
		}
	}
	for (int i = -radius; i <= radius; i++) {
		for (int j = -radius; j <= radius; j++) {
			int idx = (i + radius) * size + j + radius;
			returnPC += photo.at<Vec3b>(clamp<int>(x + j, photo.rows - 1, 0), clamp<int>(y + i, photo.cols - 1, 0))[rgb] * vector[idx];
		}
	}
	return returnPC;
}


Mat Gaussian_blur_filter(const Mat& photo, int radius, int sigma) {
	Mat result_Img;
	photo.copyTo(result_Img);
	int x, y;
	for (int x = 0; x < photo.rows; x++) {
		for (int y = 0; y < photo.cols; y++) {
			result_Img.at<Vec3b>(x, y)[0] = calculateNewPC(photo, x, y, 0, radius, sigma); //B
			result_Img.at<Vec3b>(x, y)[1] = calculateNewPC(photo, x, y, 1, radius, sigma); //G
			result_Img.at<Vec3b>(x, y)[2] = calculateNewPC(photo, x, y, 2, radius, sigma); //R
		}
	}
	return result_Img;
}
Mat MedianFilter(const Mat& photo) {
	Mat result_Img = photo.clone();
	cvtColor(result_Img, result_Img, COLOR_BGR2GRAY);
	int mRadius = 1;
	int size = 2 * mRadius + 1;
	std::vector<int> vector(size * size);
	for (int x = 0; x < photo.cols; x++)
		for (int y = 0; y < photo.rows; y++) {
			for (int i = -mRadius; i <= mRadius; i++)
				for (int j = -mRadius; j <= mRadius; j++) {
					int idx = (i + mRadius) * size + j + mRadius;
					int z = clamp<int>(x + j, result_Img.cols - 1, 0);
					int q = clamp<int>(y + i, result_Img.rows - 1, 0);
					vector[idx] = result_Img.at<uchar>(q, z);
				}
			sort(vector.begin(), vector.end());
			result_Img.at<uchar>(y, x) = vector[4];
		}
	return result_Img;
};

int main(int argc, char* argv[]) {
	Mat main_Image = imread("image.jpg");
	if (main_Image.empty()) {
		cout << "Error: the image has been incorrectly loaded." << endl;
		system("pause");
		return 0;
	}
	double bright = 0.0, coef = 50.0;

	namedWindow("DEFAULT_PICTURES");
	imshow("DEFAULT_PICTURES", main_Image);
	waitKey(0);
	cvDestroyWindow("DEFAULT_PICTURES");

	namedWindow("MODELING_GAUS_NOISE");
	Mat res;
	main_Image.copyTo(res);
	long double t1_my_gaus_noise = clock();
	Mat image_with_noise = AddGaussianNoise(res, bright, coef);
	long double t2_my_gaus_noise = clock();
	t2_my_gaus_noise -= t1_my_gaus_noise;
	cout << "Clock_of_MODELING_GAUS_NOISE: " << setprecision(15) << t2_my_gaus_noise / CLOCKS_PER_SEC << endl;
	imshow("GAUS_NOISE", image_with_noise);
	imwrite("D:gaus_noise.jpg", image_with_noise);
	waitKey(0);
	cvDestroyWindow("MODELING_GAUS_NOISE");

	namedWindow("GAUS_FILTER");
	Mat res_1;
	image_with_noise.copyTo(res_1);
	long double t1_myGaus = clock();
	Mat image_without_noise_2 = Gaussian_blur_filter(res_1, 3, 10);
	long double t2_myGaus = clock();
	t2_myGaus -= t1_myGaus;
	cout << "Clock_of_GAUS_FILTER: " << setprecision(15) << t2_myGaus / CLOCKS_PER_SEC << endl;
	imshow("GAUS_FILTER", image_without_noise_2);
	imwrite("D:gaus_filter.jpg", image_without_noise_2);
	waitKey(0);
	cvDestroyWindow("GAUS_FILTER");

	namedWindow("MEDIAN_FILTER");
	Mat res_4;
	image_with_noise.copyTo(res_4);
	long double t1_myMedian = clock();
	Mat image_without_noise_3 = MedianFilter(res_4);
	long double t2_myMedian = clock();
	t2_myMedian -= t1_myMedian;
	cout << "Clock_of_Median_FILTER: " << setprecision(15) << t2_myMedian / CLOCKS_PER_SEC << endl;
	imshow("Median_FILTER", image_without_noise_3);
	imwrite("D:median_filter.jpg", image_without_noise_3);
	waitKey(0);
	cvDestroyWindow("MEDIAN_FILTER");

	namedWindow("OPENCV_MEDIAN");
	Mat res_2;
	Mat image_without_noise;
	image_with_noise.copyTo(res_2);
	long double t1_opencv_median = clock();
	medianBlur(res_2, image_without_noise, 7);
	long double t2_opencv_median = clock();
	t2_opencv_median -= t1_opencv_median;
	cout << "Clock_of_OPENCV_MEDIAN: " << setprecision(15) << t2_opencv_median / CLOCKS_PER_SEC << endl;
	imshow("OPENCV_MEDIAN", image_without_noise);
	imwrite("D:opencv_median_filter.jpg", image_without_noise);
	waitKey(0);
	cvDestroyWindow("OPENCV_MEDIAN");

	namedWindow("OPENCV_GAUS");
	Mat res_3;
	Mat image_without_noise_1;
	image_with_noise.copyTo(res_3);
	long double t1_opencv_gaus = clock();
	GaussianBlur(res_3, image_without_noise_1, Size(5, 5), 0, 0);
	long double t2_opencv_gaus = clock();
	t2_opencv_gaus -= t1_opencv_gaus;
	cout << "Clock_of_OPENCV_GAUS: " << setprecision(15) << t2_opencv_gaus / CLOCKS_PER_SEC << endl;
	imshow("OPENCV_GAUS", image_without_noise_1);
	imwrite("D:opencv_gaus_filter.jpg", image_without_noise_1);
	waitKey(0);
	cvDestroyWindow("OPENCV_GAUS");

	system("pause");
	return 0;
}
