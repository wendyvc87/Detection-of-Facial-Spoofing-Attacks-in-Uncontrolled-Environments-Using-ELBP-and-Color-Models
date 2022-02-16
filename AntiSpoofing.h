#ifndef _AntiSpoofing_H_
#define _AntiSpoofing_H_

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
//#include <dlib/gui_widgets.h>
#include "precompilacion.h"
#include "AlgPre.h"
#include "AlgClasf.h"

using namespace cv;
using namespace std;
using namespace dlib;


class AntiSpoofing
{
private:
	cv::Mat trainingDataPO;
	cv::Mat trainingDataG;
	cv::Mat trainingDataLBP;
	cv::Mat trainingClasse;
	cv::Mat testDataPO;
	cv::Mat testDataG;
	cv::Mat testDataLBP;
	cv::Mat testClasses;
	cv::Mat trainingDataG2;
	cv::Mat trainingDataG3;
	cv::Mat testDataG2;
	cv::Mat testDataG3;
public:
	void DeteccionRostro(Mat Iin, Mat &Iout, std::vector<dlib::rectangle> &facesRect, std::vector<cv::Rect> &faces);
	void histograma(Mat input, float VectoCarac[]);
	void Entrenamiento();
	void Prediccion();
	void Preprocesamiento(Mat imgRostro, float VectoCarac[], int &retorno);
};

#endif