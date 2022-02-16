#include "lbp.h"
using namespace cv;

namespace lbp {
	template <typename _Tp> static
		inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) {
		Mat src = _src.getMat();
		_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
		Mat dst = _dst.getMat();
		dst.setTo(0);
		for (int n = 0; n<neighbors; n++) {
			float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n / static_cast<float>(neighbors));
			float y = static_cast<float>(radius) * cos(2.0*CV_PI*n / static_cast<float>(neighbors));
			int fx = static_cast<int>(floor(x));
			int fy = static_cast<int>(floor(y));
			int cx = static_cast<int>(ceil(x));
			int cy = static_cast<int>(ceil(y));
			float ty = y - fy;
			float tx = x - fx;
			float w1 = (1 - tx) * (1 - ty);
			float w2 = tx  * (1 - ty);
			float w3 = (1 - tx) *      ty;
			float w4 = tx  *      ty;
			for (int i = radius; i < src.rows - radius; i++) {
				for (int j = radius; j < src.cols - radius; j++) {
					float t = w1*src.at<_Tp>(i + fy, j + fx) + w2*src.at<_Tp>(i + fy, j + cx) + w3*src.at<_Tp>(i + cy, j + fx) + w4*src.at<_Tp>(i + cy, j + cx);
					dst.at<int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) || (std::abs(t - src.at<_Tp>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
				}
			}
		}
	}

}

void lbp::elbp(InputArray src, OutputArray dst, int radius, int neighbors) {
	switch (src.type()) {
	case CV_8SC1:   elbp_<char>(src, dst, radius, neighbors); break;
	case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
	case CV_16SC1:  elbp_<short>(src, dst, radius, neighbors); break;
	case CV_16UC1:  elbp_<unsigned short>(src, dst, radius, neighbors); break;
	case CV_32SC1:  elbp_<int>(src, dst, radius, neighbors); break;
	case CV_32FC1:  elbp_<float>(src, dst, radius, neighbors); break;
	case CV_64FC1:  elbp_<double>(src, dst, radius, neighbors); break;
	default: break;
	}
}



