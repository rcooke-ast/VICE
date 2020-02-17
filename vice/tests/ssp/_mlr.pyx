# cython: language_level = 3, boundscheck = False 

from __future__ import absolute_import 
__all__ = [
	"test_mass_lifetime_relationship" 
] 
from .._test_utils import unittest 
from . cimport _mlr 


def test_mass_lifetime_relationship(): 
	""" 
	Tests the mass lifetime relationship function implemented at 
	vice/src/ssp/mlr.h 
	""" 
	return unittest("Mass-lifetime relationship", 
		_mlr.test_main_sequence_turnoff_mass) 

