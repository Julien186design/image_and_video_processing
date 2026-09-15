#ifndef ETF_H_
#define ETF_H_

#include <opencv2/opencv.hpp>

class ETF
{
public:
    ETF();
    ETF(cv::Size);
    void initial_ETF(std::string, cv::Size);
    void refine_ETF(int kernel);

    cv::Size s;
    cv::Mat gradientMag; // Normalized gradient magnitude
    cv::Mat flowField;   // edge tangent flow
    cv::Mat refinedETF;  // ETF after refinement

private:
    cv::Mat rotate(const cv::Mat &src, double degree);
    cv::Vec3f computeNewVector(int x, int y, int kernel);
    float computePhi(const cv::Vec3f &x, const cv::Vec3f &y);
    float computeWs(const cv::Point2f &x, const cv::Point2f &y, int r);
    float computeWm(float gradmag_x, float gradmag_y);
    float computeWd(const cv::Vec3f &x, const cv::Vec3f &y);
};

#endif // ETF_H_