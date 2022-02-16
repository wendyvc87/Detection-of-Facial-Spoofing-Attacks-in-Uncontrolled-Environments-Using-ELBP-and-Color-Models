#pragma once

#ifndef _PRECOMP_H_
#define _PRECOMP_H_


#if defined MACROS_EXTENDED
#	if defined WIN32_LEAN_AND_MEAN
#		include <Windows.h>
#	else
#		include <iostream>
#	endif
#endif

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/ml/ml.hpp"
#include <opencv2/ml/ml.inl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <sstream>
#include "lbp.h"
#endif 
