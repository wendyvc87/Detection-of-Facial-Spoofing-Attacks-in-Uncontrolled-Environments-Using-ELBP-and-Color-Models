#ifndef _AlgClasf_H_
#define _AlgClasf_H_

#include "precompilacion.h"

using namespace cv;
using namespace std;

class AlgClasf
{
private:
public:
	
	void MLP(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses);
};

#endif