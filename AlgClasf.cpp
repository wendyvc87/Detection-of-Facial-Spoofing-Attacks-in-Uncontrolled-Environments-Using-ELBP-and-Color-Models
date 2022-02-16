#include "AlgClasf.h"
void AlgClasf::MLP(cv::Mat& trainingData, cv::Mat& trainingClasses, cv::Mat& testData, cv::Mat& testClasses) {
	try {
		int num_of_label = 2;
		Mat labels = Mat::zeros(trainingClasses.rows, num_of_label, CV_32FC1);
		for (int i = 0; i < trainingClasses.rows; i++) {
			int idx = trainingClasses.at<int>(i, 0);
			labels.at<float>(i, idx) = 1.0f;
		}
		cv::Mat layers = cv::Mat(3, 1, CV_32S);
		layers.row(0) = cv::Scalar(trainingData.cols);
		layers.row(1) = cv::Scalar(100);
		layers.row(2) = cv::Scalar(num_of_label);
		Ptr<ml::ANN_MLP> mlp = ml::ANN_MLP::create();
		mlp->setLayerSizes(layers);
		mlp->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001);
		mlp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1, 1);
		mlp->setBackpropMomentumScale(0.05f);
		mlp->setBackpropWeightScale(0.05f);
		mlp->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10000, 0.00001f));
		mlp->train(trainingData, ml::SampleTypes::ROW_SAMPLE, labels);

		cv::Mat TPredic;
		for (int i = 0; i < testData.rows; i++) {
			float fp = mlp->predict(testData.row(i));
			int clase = testClasses.at<int>(i, 0);
			int VPredic[2] = { clase,fp };
			//cout << clase << "-" << fp << endl;
			Mat tempPred(1, 2, CV_32SC1, VPredic);
			TPredic.push_back(tempPred);
		}
		
	}
	catch (Exception e)
	{
	}
}
