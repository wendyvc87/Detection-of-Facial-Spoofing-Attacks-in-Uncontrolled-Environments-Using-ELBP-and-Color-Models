#include "AntiSpoofing.h"
std::vector<string> imgEnts2;
std::vector<string> imgPreds2;

Mat ClassEntMats;
Mat ClassPredMats;
Mat TPredic;

int main()
{
	try {
		AntiSpoofing spoof;

		stringstream LigEntrenamiento;
		string direccionIn;
		stringstream LigPrediccion;
		string direccionOut;
		
		
		direccionIn = LigEntrenamiento.str();
		direccionOut = LigPrediccion.str();
		
		spoof.Entrenamiento();
		spoof.Prediccion();
		
	}
	catch (cv::Exception & e)
	{
		std::cout << e.msg << endl;

	}
	cvWaitKey(0);
	return 0;
}

void AntiSpoofing::Entrenamiento() {


	trainingDataG.release();
	trainingClasse.release();
	for (int n = 0; n < imgEnts2.size(); n++)
	{
		Mat imgRostro = cv::imread(imgEnts2[n]);
		int clase = ClassEntMats.at<int>(n, 0);
		int retorno=0;

		float VectoCaracHisto[256] = { 0.0 };
		if ((clase == 0 && n == 1) || (clase == 1 && n == 1500)) {
			Preprocesamiento(imgRostro, VectoCaracHisto, retorno);
		}
		else
		{
			Preprocesamiento(imgRostro, VectoCaracHisto, retorno);
		}
		if (retorno == 1)
		{
			Mat tempCar2(1, 256, CV_32F, VectoCaracHisto);
			trainingDataG.push_back(tempCar2);


			int VClase[1] = { clase };
			Mat tempClas(1, 1, CV_32SC1, VClase);
			trainingClasse.push_back(tempClas);
		}
	}
}
void AntiSpoofing::Prediccion() {
	
	try {
		AlgClasf C;
		
		testDataG.release();
		testClasses.release();
		
		for (int n = 0; n < imgPreds2.size(); n++)
		{
			
			Mat imgRostro = cv::imread(imgPreds2[n]);
			int clase = ClassPredMats.at<int>(n, 0);
			int retorno = 0;
			

			float VectoCaracHisto[256] = { 0.0 };
			Preprocesamiento(imgRostro, VectoCaracHisto, retorno);

			if (retorno == 1)
			{
				Mat tempCar2(1, 256, CV_32F, VectoCaracHisto);
				testDataG.push_back(tempCar2);

				int VClase[1] = { clase };
				Mat tempClas(1, 1, CV_32SC1, VClase);
				testClasses.push_back(tempClas);
				
			}
		}
				
		C.MLP(trainingDataG, trainingClasse, testDataG, testClasses);
		
	}
	catch (cv::Exception & e)
	{
		
	}
}
void  AntiSpoofing::Preprocesamiento(Mat imgRostro, float VectoCarac[], int &retorno) {
	
	Mat IMAGEN;
	std::vector<dlib::rectangle> facesRect;
	std::vector<cv::Rect> faces;
	DeteccionRostro(imgRostro, IMAGEN, facesRect, faces);
	if (faces.size() == 0)
	{
		retorno = 0;//No se encontro rostro
	}
	else if (facesRect.size() > 0)
	{
		retorno = 1;
		
		Mat output2, temp;
		cv::cvtColor(IMAGEN, IMAGEN, CV_BGR2YCrCb);
		output2 = IMAGEN;
		cv::cvtColor(output2, IMAGEN, CV_BGR2HSV);
		output2 = IMAGEN;
		cv::cvtColor(output2, output2, CV_BGR2GRAY);
		lbp::elbp(output2, output2, 2, 16);
		histograma(output2, VectoCarac);
	}
}
void AntiSpoofing::DeteccionRostro(Mat Iin, Mat &Iout, std::vector<dlib::rectangle> &facesRect, std::vector<cv::Rect> &faces)
{
	
	Iin.copyTo(Iout);
	
	cv::CascadeClassifier face_cascade;
	face_cascade.load("Docs/haarcascade_frontalface_alt.xml");
	const cv::Scalar scalar(255, 255, 255);
	bool rostro = false;

	face_cascade.detectMultiScale(Iin, faces, 1.1, 3, 0, cv::Size(60, 60));
	if (faces.size() != 0) {
		for (cv::Rect& rc : faces) {
			if (((rc.width > 100 && rc.height > 100) || faces.size() == 1) && !rostro) {
				rostro = true;
				int x=0, y=0, alto=0, ancho = 0;
				x = rc.x + 20;
				if (rc.y < 20)
					y = 0;
				else
					y = rc.y - 20;

				if (rc.height < 30)
					alto = 0;
				else
					alto = rc.height - 30;
				if ((rc.y + rc.width + 40) > Iout.rows)
					ancho = Iout.rows - rc.y;
				else
					ancho = rc.width + 40;

				Point p0(x, y);
				Rect zona(p0, Size(alto, ancho));
				Iout = Iout(zona);
				cv::resize(Iout, Iout, cv::Size(300, 350));

				Mat im2(Iout.rows, Iout.cols, CV_8UC1, Scalar(0, 0, 0));
				ellipse(im2, Point(150, 175), Size(150, 175), 0, 0, 360, Scalar(255, 255, 255), -1, 8);

				cv::Mat r = cv::Mat::zeros(Iout.size(), Iout.type());
				Iout.copyTo(r, im2);

				r.copyTo(Iout);
				facesRect.push_back(dlib::rectangle(0, 0, 300, 350));
			}
		}
	}
}
void AntiSpoofing::histograma(Mat input, float VectoCarac[])
{
	for (int i = 0; i < 256; i++)
	{
		VectoCarac[i] = 0;
	}
	for (int r = 0; r < input.rows; r++)
	{
		for (int c = 0; c < input.cols; c++)
		{
			int ind = input.at < unsigned char >(r, c);
			VectoCarac[ind] = VectoCarac[ind] + 1;
		}
	}
}
