# cython: language_level = 3, boundscheck = False 

from __future__ import absolute_import 
from ._objects cimport CCSNE_YIELD_SPECS 

cdef extern from "../src/ccsne.h": 
	double CC_YIELD_STEP 
	double CC_YIELD_GRID_MIN 
	double CC_YIELD_GRID_MAX 
	double CC_MIN_STELLAR_MASS 
	CCSNE_YIELD_SPECS *ccsne_yield_initialize() 
	void ccsne_yield_free(CCSNE_YIELD_SPECS *ccsne_yield) 
	double *IMFintegrated_fractional_yield_numerator(char *file, char *IMF, 
		double m_lower, double m_upper, double tolerance, char *method, 
		unsigned long Nmax, unsigned long Nmin) 
	double *IMFintegrated_fractional_yield_denominator(char *IMF, 
		double m_lower, double m_upper, double tolerance, char *method, 
		unsigned long Nmax, unsigned long Nmin)
