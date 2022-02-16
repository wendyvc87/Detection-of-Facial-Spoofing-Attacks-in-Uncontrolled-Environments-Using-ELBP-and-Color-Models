#ifndef __lbp_H__
#define __lbp_H__
#include "precompilacion.h"
using namespace cv;

namespace lbp {
	void elbp(InputArray src, OutputArray dst, int radius = 1, int neighbors = 8);
}

#endif