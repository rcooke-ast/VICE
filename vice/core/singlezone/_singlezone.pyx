# cython: language_level = 3, boundscheck = False
""" 
This file implements the wrapper for the singlezone object, which runs 
simulations under the onezone approximation. Most of the subroutines are in C, 
found in the vice/src/ directory within the root tree. 
""" 

# Python imports 
from __future__ import absolute_import 
from ...version import version as _VERSION_ 
from ..._globals import _RECOGNIZED_ELEMENTS_ 
from ..._globals import _RECOGNIZED_IMFS_ 
from ..._globals import _VERSION_ERROR_ 
from ..._globals import _DEFAULT_FUNC_ 
from ..._globals import _DEFAULT_BINS_ 
from ..._globals import _DIRECTORY_ 
from ..._globals import VisibleDeprecationWarning 
from ..._globals import ScienceWarning 
from ..dataframe import evolutionary_settings 
from ..dataframe import atomic_number 
from ..dataframe import primordial 
from ..dataframe import solar_z 
from ..dataframe import sources 
from ..dataframe import base 
from ..outputs import output 
from ...yields import agb 
from ...yields import ccsne 
from ...yields import sneia 
from .. import _pyutils 
import math as m 
import warnings 
import numbers 
import pickle 
import sys 
import os 
if sys.version_info[:2] == (2, 7): 
	strcomp = basestring 
	input = raw_input 
elif sys.version_info[:2] >= (3, 5): 
	strcomp = str 
else: 
	_VERSION_ERROR_() 
try: 
	ModuleNotFoundError 
except NameError: 
	ModuleNotFoundError = ImportError 
try: 
	"""
	dill extends the pickle module and allows functional attributes to be 
	encoded. In later versions of python 3, dill.dump must be called instead 
	of pickle.dump. All cases can be taken care of by overriding the native 
	pickle module and letting dill masquerade as pickle. 
	""" 
	import dill as pickle 
except (ModuleNotFoundError, ImportError): 
	pass 

# C imports 
from libc.stdlib cimport malloc 
from libc.string cimport strlen 
from . cimport _agb 
from .._cutils cimport set_string 
from .._cutils cimport copy_pylist 
from .._cutils cimport setup_imf 
from .._cutils cimport map_ccsne_yield 
from .._cutils cimport map_sneia_yield 
from .._cutils cimport map_agb_yield 
from .._cutils cimport copy_2Dpylist 
from . cimport _element 
from . cimport _io 
from . cimport _singlezone 
from . cimport _sneia 
from . cimport _ssp 

_RECOGNIZED_MODES_ = tuple(["ifr", "sfr", "gas"]) 
_RECOGNIZED_DTDS_ = tuple(["exp", "plaw"]) 

""" 
NOTES 
===== 
cdef class objects do not transfer the docstrings of class attributes to the 
compiled output, leaving out the internal documentation. For this reason, 
wrapping of the singlezone object has two layers -> a python class and a 
C class. In the python class, there is only one attribute: the C version of 
the wrapper. The docstrings are written here, and each function/setter 
only calls the C version of the wrapper. While this is a more complicated 
wrapper, it preserves the internal documentation. In order to maximize 
readability, the setter functions of the C version of the wrapper have brief 
notes on the physical interpretation of each attribute as well as the allowed 
types and values. 
""" 

#--------------------------- SINGLEZONE C VERSION ---------------------------# 
cdef class c_singlezone: 

	""" 
	Wrapping of the C version of the singlezone object 
	""" 

	# cdef SINGLEZONE *_sz 
	# cdef object _func 
	# cdef object _imf 
	# cdef object _eta 
	# cdef object _enhancement 
	# cdef zone_entrainment _entrainment 
	# cdef object _tau_star 
	# cdef object _zin 
	# cdef object _ria 
	# cdef double _Mg0 
	# cdef object _agb_model 

	def __cinit__(self): 
		self._sz = _singlezone.singlezone_initialize()

	def __init__(self, 
		name = "onezonemodel", 
		func = _DEFAULT_FUNC_, 
		mode = "ifr", 
		verbose = False, 
		elements = ("fe", "sr", "o"), 
		IMF = "kroupa", 
		eta = 2.5, 
		enhancement = 1, 
		Zin = 0, 
		recycling = "continuous", 
		bins = _DEFAULT_BINS_, 
		delay = 0.15, 
		RIa = "plaw", 
		Mg0 = 6.0e9, 
		smoothing = 0, 
		tau_ia = 1.5, 
		tau_star = 2.0, 
		dt = 0.01, 
		schmidt = False, 
		MgSchmidt = 6.0e9, 
		schmidt_index = 0.5, 
		m_upper = 100, 
		m_lower = 0.08, 
		postMS = 0.1, 
		Z_solar = 0.014, 
		agb_model = None): 

		"""
		All properties may be specified via __init__ as a keyword. 
		""" 
		# ------------------------------------------------------------------- # 
		"""
		Initialize the C singlezone object and call the python setter 
		functions. Most of the attributes are stored directly in C. 
		"""  
		self.name = name 
		self.func = func 
		self.mode = mode 
		self.verbose = verbose 
		self.elements = elements 
		self.IMF = IMF 
		self.eta = eta 
		self.enhancement = enhancement 
		self._entrainment = zone_entrainment() 
		self.Zin = Zin 
		self.recycling = recycling 
		self.bins = bins 
		self.delay = delay 
		self.RIa = RIa 
		self.Mg0 = Mg0 
		self.smoothing = smoothing 
		self.tau_ia = tau_ia 
		self.tau_star = tau_star 
		self.dt = dt 
		self.schmidt = schmidt 
		self.MgSchmidt = MgSchmidt 
		self.schmidt_index = schmidt_index 
		self.m_upper = m_upper 
		self.m_lower = m_lower 
		self.postMS = postMS 
		self.Z_solar = Z_solar 
		self.agb_model = agb_model 

	def __dealloc__(self): 
		_singlezone.singlezone_free(self._sz) 

	def object_address(self): 
		""" 
		Returns the memory address of the associated SINGLEZONE struct in C. 
		""" 
		return _singlezone.singlezone_address(self._sz) 

	@property
	def name(self):  
		# docstring in python version 
		return "".join([chr(self._sz[0].name[i]) for i in range(
			strlen(self._sz[0].name))])[:-5] 

	@name.setter
	def name(self, value): 
		""" 
		Name of the simulation, also the directory that the output is written 
		to. 

		Allowed Types 
		============= 
		str 

		Allowed Values 
		============== 
		Simple strings, or those of the format 'path/to/dir' 

		All values will pass the setter except for empty strings. Those that 
		are not valid directory names will fail at runtime when self.run() is 
		called. 
		""" 
		if isinstance(value, strcomp): 
			if _pyutils.is_ascii(value): 
				if len(value) == 0: 
					raise ValueError("""Attribute 'name' must not be an \
empty string.""") 
				else: 
					pass 
				while value[-1] == '/': 
					# remove any '/' that the user puts on 
					value = value[:-1] 
				if value.lower().endswith(".vice"): 
					# force the .vice extension to lower-case 
					value = "%s.vice" % (value[:-5]) 
				else: 
					value = "%s.vice" % (value) 
				set_string(self._sz[0].name, value) 
			else: 
				raise ValueError("String must be ascii. Got: %s" % (
					value))
		else: 
			raise TypeError("Attribute 'name' must be of type str. Got: %s" % (
				type(value))) 

	@property 
	def func(self): 
		# docstring in python version 
		return self._func

	@func.setter
	def func(self, value): 
		""" 
		Specified function of time, interpretation set by attribute mode 

		Allowed Types 
		============= 
		function 

		Allowed Values 
		============== 
		Accepting one parameter, interpreted as time in Gyr 
		""" 
		if callable(value): 
			"""
			The function by design must be called. It must also take only 
			one parameter. 
			""" 
			_pyutils.args(value, """\
Attribute 'func' must accept one numerical parameter.""")
			self._func = value 
		else: 
			raise TypeError("""Attribute func must be of type <function>. \
Got: %s""" % (type(value))) 

	@property
	def mode(self): 
		# docstring in python version 
		return "".join([chr(self._sz[0].ism[0].mode[i]) for i in range(
			strlen(self._sz[0].ism[0].mode))]) 

	@mode.setter 
	def mode(self, value): 
		""" 
		Specification of func 

		Allowed Types 
		============= 
		str [case-insensitive] 

		Allowed Values 
		============== 
		"gas", "sfr", "ifr" 
		""" 
		if isinstance(value, strcomp): 
			if value.lower() in _RECOGNIZED_MODES_: 
				set_string(self._sz[0].ism[0].mode, value.lower()) 
			else: 
				raise ValueError("Unrecognized mode: %s" % (value)) 
		else: 
			raise TypeError("Attribute 'mode' must be of type str. Got: %s" % (
				type(value))) 

	@property 
	def verbose(self): 
		# docstring in python version 
		return bool(self._sz[0].verbose) 

	@verbose.setter 
	def verbose(self, value): 
		""" 
		Whether or not to print the time as the simulation evolves. 

		Allowed Types 
		============= 
		bool 

		Allowed Values 
		============== 
		True and False 
		""" 
		if isinstance(value, numbers.Number) or isinstance(value, bool): 
			if value: 
				self._sz[0].verbose = 1 
			else: 
				self._sz[0].verbose = 0 
		else: 
			raise TypeError("""Attribute 'verbose' must be interpretable as \
a boolean. Got: %s""" % (type(value))) 

	@property
	def elements(self): 
		# docstring in python version 
		elements = self._sz[0].n_elements * [None] 
		for i in range(self._sz[0].n_elements): 
			elements[i] = "".join(
				[chr(self._sz[0].elements[i][0].symbol[j]) for j in range(
					strlen(self._sz[0].elements[i][0].symbol))] 
			) 
		return tuple(elements[:]) 

	@elements.setter 
	def elements(self, value): 
		""" 
		Symbols of elements to track 

		Allowed Types 
		============= 
		array-like 

		Allowed Values 
		============== 
		Those in _RECOGNIZED_ELEMENTS_ 
		
		This setter function is less complicated than its length would 
		suggest. A lot of other attributes to the singlezone object are 
		dependent on the elements being simulated, and so are dependent on the 
		current setting. Most of the length to this function takes this into 
		account, while checking for the presence of other attributes to not 
		break the first initialization of the singlezone object. 
		""" 
		# Must be an array like object of strings 
		value = _pyutils.copy_array_like_object(value) 
		if all(map(lambda x: isinstance(x, strcomp), value)): 
			if all(map(lambda x: x.lower() in _RECOGNIZED_ELEMENTS_, value)): 

				# First check for a conflict w/the AGB yield grid 
				if not hasattr(self, "agb_model"): 
					# But don't break the first initialization of the object 
					pass 
				if (any(map(lambda x: atomic_number[x] > 28, value)) and 
					self._agb_model == "karakas10"): 
					raise LookupError("""\
The Karakas (2010), MNRAS, 403, 1413 study did not report yields for elements \
heavier than nickel. Please modify the attribute 'elements' to exclude these \
elements from the simulation (or adopt an alternate yield model) before 
proceeding.""") 
				else: 
					pass 

				# if we need to reassign old SNe Ia settings 
				assign_ia_params = False 
				# Free an existing set of elements 
				if self._sz[0].elements is not NULL: 
					"""
					grab a copy of the minimum SNe Ia delay time, tau_ia 
					timescales, and DTD to not break those features -> only 
					need one, they're all the same 
					"""
					x = self._sz[0].elements[0][0].sneia_yields[0].t_d 
					y = self._sz[0].elements[0][0].sneia_yields[0].tau_ia 
					assign_ia_params = True 
					for i in range(self._sz[0].n_elements): 
						_element.element_free(self._sz[0].elements[i]) 
				else: 
					pass 

				# remove duplicate elemental symbols 
				value = list(dict.fromkeys(value)) 

				# allocate memory and copy the symbols over 
				self._sz[0].n_elements = len(value) 
				self._sz[0].elements = <_element.ELEMENT **> malloc (
					self._sz[0].n_elements * sizeof(_element.ELEMENT *)) 
				for i in range(self._sz[0].n_elements): 
					self._sz[0].elements[i] = _element.element_initialize() 
					set_string(self._sz[0].elements[i][0].symbol, 
						value[i].lower()) 
					if assign_ia_params: 
						self._sz[0].elements[i][0].sneia_yields[0].t_d = x 
						self._sz[0].elements[i][0].sneia_yields[0].tau_ia = y 
						if callable(self._ria): 
							set_string(
								self._sz[0].elements[i][0].sneia_yields[0].dtd, 
								"custom") 
						else: 
							set_string(
								self._sz[0].elements[i][0].sneia_yields[0].dtd, 
								self._ria) 
					else: 
						pass 
			else: 
				# Raise the error with all of the unrecognized elements 
				raise ValueError("Unrecognized element(s): %s" % (
					list(filter(lambda x: x not in _RECOGNIZED_ELEMENTS_, 
						value)))) 
		else: 
			raise TypeError("""Attribute 'element' must be of type str. \
Got: %s""" % (type(
				# Raise the error with type of the first offending element 
				list(filter(lambda x: not isinstance(x, strcomp), value))[0] 
				)))

		"""
		Under the hood, if a new list of elements was passed and infall 
		metallicity settings are a VICE dataframe, default each element's 
		infall metallicities to zero if their settings aren't already there. 
		""" 
		if hasattr(self, "Zin") and isinstance(self._zin, 
			evolutionary_settings): 
			for i in self.elements: 
				if i not in self._zin.keys(): 
					self._zin[i.lower()] = 0.0 
				else: 
					continue 
		else: 
			# object just now being initialized 
			pass 

	@property
	def IMF(self): 
		# docstring in python version 
		return self._imf 

	@IMF.setter 
	def IMF(self, value): 
		""" 
		Stellar initial mass function (IMF) 

		Allowed Types 
		============= 
		str [case-insensitive] or <function> 

		Allowed Values 
		============== 
		"kroupa", "salpeter", or any callable function accepting one numerical 
		parameter 
		""" 
		if callable(value): 
			_pyutils.args(value, """\
Attribute 'IMF' must accept only one numerical parameter.""") 
			self._imf = value 
		elif isinstance(value, strcomp): 
			if value.lower() in _RECOGNIZED_IMFS_: 
				# set_string(self._sz[0].ssp[0].imf, value.lower()) 
				self._imf = value.lower() 
			else: 
				raise ValueError("Unrecognized IMF: %s" % (value)) 
		else: 
			raise TypeError("""Attribute 'IMF' must be of type str or a \
callable function. Got: %s""" % (type(value))) 

	@property
	def eta(self): 
		# docstring in python version 
		return self._eta 

	@eta.setter 
	def eta(self, value): 
		""" 
		Mass loading factor 

		Allowed Types 
		============= 
		function, real number 

		Allowed Values 
		============== 
		function :: accepting one parameter, interpreted as time in Gyr 
		real number :: >= 0 
		""" 
		if callable(value): 
			# make sure it takes only one parameter 
			_pyutils.args(value, """Attribute 'eta', when callable, must \
accept only one numerical parameter.""")
			self._eta = value 
		elif isinstance(value, numbers.Number): 
			if value > 0: 
				self._eta = float(value) 
			elif value == 0: 
				self._eta = 0.0 
				warnings.warn("""\
Closed-box GCE models have been shown to overpredict the metallicities of \
solar neighborhood stars. This was known as the G-dwarf problem (Tinsley \
1980, Fundamentals of Cosmic Phys., 5, 287). Outflows have been shown to be \
necessary for maintaining long-term chemical equilibrium (Dalcanton 2007, \
ApJ, 658, 941).""", ScienceWarning) 
			else: 
				raise ValueError("""Attribute 'eta' must be non-negative. \
Got: %g""" % (value))  
		else: 
			raise TypeError("""Attribute 'eta' must be either a callable \
function or a numerical value. Got: %s""" % (type(value))) 

	@property
	def enhancement(self): 
		# docstring in python version 
		return self._enhancement 

	@enhancement.setter 
	def enhancement(self, value): 
		""" 
		Ratio of outflow to ISM metallicity 

		Allowed Types 
		============= 
		function, number 

		Allowed Values 
		============== 
		function :: accepting one parameter, interpreted as time in Gyr 
		number :: 0 <= x <= 1 
		""" 
		if callable(value): 
			# make sure it only takes one parameter 
			_pyutils.args(value, """Attribute 'enhancement', when callable, \
must accept only one numerical parameter.""") 
			self._enhancement = value 
		elif isinstance(value, numbers.Number): 
			if value >= 0: 
				self._enhancement = float(value) 
			else: 
				raise ValueError("""Attribute 'enhancement' must be \
non-negative. Got: %g""" % (value)) 
		else: 
			raise TypeError("""Attribute 'enhancement' must be either a \
callable function or a numerical value. Got: %s""" % (type(value))) 

	@property 
	def entrainment(self): 
		# docstring in python version 
		return self._entrainment 

	@property
	def Zin(self): 
		# docstring in python version 
		return self._zin 

	@Zin.setter 
	def Zin(self, value): 
		""" 
		Inflow metallicity prescription 

		Allowed Types 
		============= 
		real number, function, dict/dataframe 

		Allowed Values 
		============== 
		real number :: 0 <= x <= 1 
		function :: accepting one parameter, interpreted as time in Gyr 
		dict/dataframe :: having elemental symbols as keys [case-insensitive] 
		""" 
		# Don't allow infs or nans with numerical values 
		if (isinstance(value, numbers.Number) and not m.isinf(value) and not 
			m.isnan(value)): 
			if 0 <= value <= 1: 
				self._zin = float(value) 
			else: 
				raise ValueError("""Attribute 'zin' must be non-negative. \
Got: %g""" % (value)) 
		elif isinstance(value, dict) or isinstance(value, base): 
			"""
			put everything into one dictionary then cast as an 
			evolutionary_settings dataframe. 
			""" 
			frame = {} 
			# VICE dataframes will always pass the following test 
			if not all(map(lambda x: isinstance(x, strcomp), value.keys())): 
				raise TypeError("""When initializing attribute 'Zin' as a \
dataframe, it must take only strings as keys.""") 
			else: 
				copy = dict(zip([i.lower() for i in value.keys()], 
					[value[i] for i in value.keys()])) 


			# Take each number and function from the passed dataframe 
			for i in value.keys(): 
				if i.lower() not in _RECOGNIZED_ELEMENTS_: 
					raise ValueError("Unrecognized element: %s" % (i)) 
				else: 
					if (isinstance(copy[i.lower()], numbers.Number) or 
						callable(copy[i.lower()])): 

						frame[i.lower()] = copy[i.lower()] 
					else: 
						raise TypeError("""Infall metallicity must be either \
a numerical value or a callable function. Got: %s""" % (type(copy[i.lower()]))) 
			
			# any tracked elements missed by the passed value 
			for i in self.elements: 
				if i.lower() not in value.keys(): 
					if isinstance(self._zin, evolutionary_settings): 
						frame[i.lower()] = self._zin[i.lower()] 
					elif (isinstance(self._zin, numbers.Number) or 
						callable(self._zin)): 
						frame[i.lower()] = self._zin 
					elif self._zin == None: 
						# not initialized yet 
						frame[i.lower()] = 0.0 
					else: 
						raise SystemError("Internal Error") 
				else: 
					continue 

			self._zin = evolutionary_settings(frame, "Infall metallicity") 
		elif callable(value): 
			_pyutils.args(value, """Attribute 'zin', when callable, must take \
only one numerical parameter.""") 
			self._zin = value 
		else: 
			raise TypeError("""\
Attribute 'Zin' must be either a callable function of time, a numerical \
value, or a dictionary/VICE dataframe containing any combination thereof for \
individual elements. Got: %s""" % (type(value)))  

	@property
	def recycling(self): 
		# docstring in python version 
		if self._sz[0].ssp[0].continuous: 
			return "continuous" 
		else: 
			return self._sz[0].ssp[0].R0 

	@recycling.setter 
	def recycling(self, value): 
		""" 
		Recycling prescription 

		Allowed Types 
		============= 
		str [case-insensitive], real number 

		Allowed Values 
		============== 
		str :: "recycling" 
		real number :: 0 < x < 1 
		""" 
		if isinstance(value, numbers.Number): 
			if 0 <= value <= 1: 
				self.__recycling_warnings(value)
				self._sz[0].ssp[0].R0 = float(value)  
				self._sz[0].ssp[0].continuous = 0 
			else: 
				raise ValueError("""The cumulative return fraction must be \
between 0 and 1 to be physical. Got: %g""" % (value)) 
		elif isinstance(value, strcomp): 
			if value.lower() == "continuous": 
				self._sz[0].ssp[0].R0 = 0.0 
				self._sz[0].ssp[0].continuous = 1 
			else: 
				raise ValueError("""If attribute 'recycling' is to be a \
string, it be 'continuous' (case-insensitive). Got: %s""" % (value)) 
		else: 
			raise TypeError("""Attribute 'recycling' must be either a \
numerical value between 0 and 1 or the string 'continuous' \
(case-insensitive). Got: %s""" % (type(value)))  

	def __recycling_warnings(self, value): 
		""" 
		Raises a ScienceWarning if the instantaneous recycling parameter is 
		significantly different from that derived in Weinberg et al. (2017), 
		ApJ, 837, 183. 
		""" 
		if isinstance(value, numbers.Number) and 0 <= value <= 1: 
			# This function should only be called if this condition is met 
			if self.IMF == "kroupa": 
				if value <= 0.36 or value >= 0.44: 
					warnings.warn("""\
Weinberg, Andrews & Freudenburg (2017), ApJ, 837, 183 recommends an \
instantaneous recycling parameter of r = 0.4 for the Kroupa (2001) IMF. Got \
value with a >10%% discrepancy from this value: %g""" % (value), 
						ScienceWarning) 
				else: 
					pass 
			elif self.IMF == "salpeter": 
				if value <= 0.18 or value >= 0.22: 
					warnings.warn("""\
Weinberg, Andrews & Freudenburg (2017), ApJ, 837, 183 recommends an \
instantaneous recycling parameter of r = 0.2 for the Salpeter (1955) IMF. Got \
value with a >10%% discrepancy from this value: %g""" % (value), 
						ScienceWarning)
				else: 
					pass 
			else: 
				# callable function -> custom IMF 
				pass 
		else: 
			# This function shouldn't have even been called 
			raise SystemError("Internal Error") 

	@property
	def bins(self): 
		# docstring in python version 
		return [self._sz[0].mdf[0].bins[i] for i in range(
			self._sz[0].mdf[0].n_bins + 1)] 

	@bins.setter 
	def bins(self, value): 
		""" 
		Bins into which all MDFs will be sorted 

		Allowed Types 
		============= 
		array-like 

		Allowed Values 
		============== 
		All whose elements are real numbers 
		""" 
		value = _pyutils.copy_array_like_object(value) 
		_pyutils.numeric_check(value, TypeError, """Attribute 'bins' must \
contain only numerical values.""") 
		value = sorted(value) 	# ascending order 
		self._sz[0].mdf[0].n_bins = len(value) - 1 
		self._sz[0].mdf[0].bins = copy_pylist(value) 

	@property
	def delay(self): 
		# docstring in python version 
		return self._sz[0].elements[0][0].sneia_yields[0].t_d 

	@delay.setter 
	def delay(self, value): 
		""" 
		Minimum delay time for SNe Ia in Gyr 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		>= 0 
		""" 
		if isinstance(value, numbers.Number): 
			if value >= 0: 
				"""
				Each element has a copy of the minimum delay time in 
				their sneia_yields object 
				""" 
				for i in range(self._sz[0].n_elements): 
					self._sz[0].elements[i][0].sneia_yields[0].t_d = value 
			else: 
				raise ValueError("""Attribute 'delay' must be non-negative. \
Got: %g""" % (value))  
		else: 
			raise TypeError("""Attribute 'delay' must be a numerical value. \
Got: %s""" % (value)) 

	@property
	def RIa(self): 
		# docstring in python version 
		return self._ria 

	@RIa.setter 
	def RIa(self, value): 
		""" 
		The delay-time distribution for SNe Ia to adopt 

		Allowed Types 
		============= 
		str [case-insensitive], function 

		Allowed Values 
		============== 
		str :: "exp", "plaw" 
		function :: accepting one parameter, interpreted as time in Gyr 
		""" 
		if callable(value): 
			""" 
			Allow functionality for user-specified delay-time distributions. 
			As usual, the specified function must take only one parameter: 
			time in Gyr. 
			""" 
			_pyutils.args(value, """When callable, attribute 'RIa' must take \
only one numerical parameter.""") 
			self._ria = value 
			for i in range(self._sz[0].n_elements): 
				set_string(
					self._sz[0].elements[i][0].sneia_yields[0].dtd, "custom") 
		elif isinstance(value, strcomp): 
			if value.lower() in _RECOGNIZED_DTDS_: 
				self._ria = value 
				for i in range(self._sz[0].n_elements): 
					set_string(
						self._sz[0].elements[i][0].sneia_yields[0].dtd, 
						value.lower()) 
			else: 
				raise ValueError("Unrecognized SNe Ia DTD: %s" % (value)) 
		else: 
			raise TypeError("""Attribute 'dtd' must be either a callable \
function or a type string denoting a built-in delay-time distribution. \
Got: %s""" % (type(value)))  

	@property
	def Mg0(self): 
		# docstring in python version 
		return self._Mg0 

	@Mg0.setter 
	def Mg0(self, value): 
		""" 
		Initial gas supply in Msun. Only relevant when mode == 'ifr' 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		>= 0 
		""" 
		if isinstance(value, numbers.Number): 
			if value >= 0: 
				self._Mg0 = float(value) 
				# avoid ZeroDivisionError 
				if self._Mg0 == 0: self._Mg0 += 1e-12 
			else: 
				raise ValueError("""Attribute 'Mg0' must be non-negative. \
Got: %s""" % (value)) 
		else: 
			raise TypeError("""Attribute 'Mg0' must be a numerical value. \
Got: %s""" % (type(value))) 

	@property
	def smoothing(self): 
		# docstring in python version 
		return self._sz[0].ism[0].smoothing_time 

	@smoothing.setter 
	def smoothing(self, value): 
		""" 
		Smoothing timescale in Gyr 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		> 0 
		""" 
		if isinstance(value, numbers.Number): 
			if value >= 0: 
				self._sz[0].ism[0].smoothing_time = value 
			else: 
				raise ValueError("""Attribute 'smoothing' must be \
non-negative. Got: %g""" % (value)) 
		else: 
			raise TypeError("""Attribute 'smoothing' must be a numerical \
value. Got: %s""" % (type(value))) 

	@property
	def tau_ia(self): 
		# docstring in python version 
		return self._sz[0].elements[0][0].sneia_yields[0].tau_ia 

	@tau_ia.setter 
	def tau_ia(self, value): 
		""" 
		E-folding timescale of SNe Ia when RIa == "exp" 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		> 0 
		""" 
		if isinstance(value, numbers.Number): 
			if value > 0: 
				""" 
				Each element has a copy of the minimum delay time in their 
				sneia_yields object 
				""" 
				for i in range(self._sz[0].n_elements): 
					self._sz[0].elements[i][0].sneia_yields[0].tau_ia = value 
			else: 
				raise ValueError("""Attribute 'tau_ia' must be a positive. \
Got: %s""" % (value)) 
		else: 
			raise TypeError("""Attribute 'tau_ia' must be a numerical \
value. Got: %s""" % (type(value))) 

	@property
	def tau_star(self): 
		# docstring in python version 
		return self._tau_star 

	@tau_star.setter 
	def tau_star(self, value): 
		""" 
		Star formation efficiency timescale in Gyr. 

		Allowed Types 
		============= 
		real number, function 

		Allowed Values 
		============== 
		real number :: > 0 
		function :: accepting on parameter, interpreted as time in Gyr 
		""" 
		if isinstance(value, numbers.Number): 
			if value > 0: 
				self._tau_star = float(value) 
			else: 
				raise ValueError("""Attribute 'tau_star' must be positive. \
Got: %g""" % (value)) 
		elif callable(value): 
			# make sure it takes only one parameter 
			_pyutils.args(value, """Attribute 'tau_star', when callable, must \
take only one numerical parameter.""") 
			self._tau_star = value 
		else: 
			raise TypeError("""Attribute 'tau_star' must be either a \
numerical value or a callable function.""") 

	@property 
	def dt(self): 
		# docstring in python version 
		return self._sz[0].dt 

	@dt.setter 
	def dt(self, value): 
		""" 
		Timestep size in Gyr 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		> 0 
		""" 
		if isinstance(value, numbers.Number): 
			if value > 0: 
				self._sz[0].dt = value 
			else: 
				raise ValueError("Attribute 'dt' must be positive definite.") 
		else: 
			raise TypeError("""Attribute 'dt' must be a numerical value. \
Got: %s""" % (type(value)))  

	@property
	def schmidt(self): 
		# docstring in python version 
		return bool(self._sz[0].ism[0].schmidt) 

	@schmidt.setter 
	def schmidt(self, value): 
		""" 
		Whether or not to adopt Kennicutt-Schmidt Law driven star formation 
		efficiency 

		Allowed Types 
		============= 
		bool (or number, will be interpreted as bool) 

		Allowed values 
		============== 
		True, False 
		""" 
		if isinstance(value, numbers.Number) or isinstance(value, bool): 
			if value: 
				self._sz[0].ism[0].schmidt = 1 
			else: 
				self._sz[0].ism[0].schmidt = 0 
		else: 
			raise TypeError("""Attribute 'schmidt' must be interpretable as \
a boolean. Got: %s""" % (type(value))) 

	@property
	def schmidt_index(self): 
		# docstring in python version 
		return self._sz[0].ism[0].schmidt_index 

	@schmidt_index.setter 
	def schmidt_index(self, value): 
		""" 
		Power-law index on Kennicutt-Schmidt Law 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		all 
		""" 
		if isinstance(value, numbers.Number): 
			self._sz[0].ism[0].schmidt_index = value 
		else: 
			raise TypeError("""Attribute 'schmidt_index' must be a numerical \
value. Got: %s""" % (type(value))) 

		""" 
		A negative power law index for the Kennicutt-Schmidt law isn't 
		necessarily unphysical, but definitely unrealistic. Warn the user if 
		they passed one in case it's a bug in their code. 
		""" 
		if self._sz[0].ism[0].schmidt_index < 0: 
			warnings.warn("""\
Attribute 'schmidt_index' is now a negative value. This may introduce \
numerical artifacts.""", ScienceWarning) 
		else: 
			pass 

	@property
	def MgSchmidt(self): 
		# docstring in python version 
		return self._sz[0].ism[0].mgschmidt 

	@MgSchmidt.setter 
	def MgSchmidt(self, value): 
		""" 
		Normalization of Kennicutt-Schmidt Law in Msun 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		> 0 
		""" 
		if isinstance(value, numbers.Number): 
			if value > 0: 
				self._sz[0].ism[0].mgschmidt = value 
			else: 
				raise ValueError("""Attribute 'MgSchmidt' must be positive \
definite. Got: %g""" % (value)) 
		else: 
			raise TypeError("""Attribute 'MgSchmidt' must be a numerical \
value. Got: %s""" % (type(value))) 

	@property
	def m_upper(self): 
		# docstring in python version 
		return self._sz[0].ssp[0].imf[0].m_upper 

	@m_upper.setter 
	def m_upper(self, value): 
		""" 
		Upper mass limit on star formation in Msun 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		> 0 
		> m_lower 
		""" 
		if isinstance(value, numbers.Number): 
			if value > 0: 
				self._sz[0].ssp[0].imf[0].m_upper = value 
			else: 
				raise ValueError("""Attribute 'm_upper' must be positive. \
Got: %s""" % (value)) 
		else: 
			raise TypeError("""Attribute 'm_upper' must be a numerical \
value. Got: %s""" % (type(value))) 

		# Raise a ScienceWarning for upper mass limits below 80 Msun. 
		if self._sz[0].ssp[0].imf[0].m_upper < 80: 
			warnings.warn("""This is a low upper mass limit on star \
formation: %g. This may introduce numerical artifacts.""" % (
				self._sz[0].ssp[0].imf[0].m_upper), ScienceWarning) 
		else: 
			pass 

	@property
	def m_lower(self): 
		# docstring in python class 
		return self._sz[0].ssp[0].imf[0].m_lower 

	@m_lower.setter 
	def m_lower(self, value): 
		""" 
		Lower mass limit on star formation in Msun 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		> 0 
		< m_upper 
		""" 
		if isinstance(value, numbers.Number): 
			if value > 0: 
				self._sz[0].ssp[0].imf[0].m_lower = value 
			else: 
				raise ValueError("""Attribute 'm_lower' must be positive. \
Got: %s""" % (value)) 
		else: 
			raise TypeError("""Attribute 'm_lower' must be a numerical \
value. Got: %s""" % (type(value))) 

		# Raise a ScienceWarning for lower mass limit above 0.2 Msun 
		if self._sz[0].ssp[0].imf[0].m_lower > 0.2: 
			warnings.warn("""This is a high lower mass limit on star \
formation: %g. This may introduce numerical artifacts.""" % (
				self._sz[0].ssp[0].imf[0].m_lower), ScienceWarning) 
		else: 
			pass 

	@property 
	def postMS(self): 
		# docstring in python class 
		return self._sz[0].ssp[0].postMS 

	@postMS.setter 
	def postMS(self, value): 
		""" 
		Ratio of a star's post main sequence lifetime to its main sequence 
		lifetime 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		> 0 and < 1 
		""" 
		if isinstance(value, numbers.Number): 
			if 0 <= value <= 1: 
				self._sz[0].ssp[0].postMS = value 
			else: 
				raise ValueError("Attribute 'postMS' must be between 0 and 1.") 
		else: 
			raise TypeError("""Attribute 'postMS' must be a numerical value. \
Got: %s""" % (type(value))) 

	@property
	def Z_solar(self): 
		# docstring in python class 
		return self._sz[0].Z_solar 

	@Z_solar.setter 
	def Z_solar(self, value): 
		""" 
		Adopted solar metallicity by mass 

		Allowed Types 
		============= 
		real number 

		Allowed Values 
		============== 
		0 < x < 1 
		""" 
		if isinstance(value, numbers.Number): 
			# This value is a mass fraction -> must be between 0 and 1
			if 0 < value < 1: 
				self._sz[0].Z_solar = value 
			else: 
				raise ValueError("""Attribute 'Z_solar' must be between 0 \
and 1. Got: %g""" % (value)) 
		else: 
			raise TypeError("""Attribute 'Z_solar' must be a numerical value. \
Got: %s""" % (type(value))) 

		"""
		ScienceWarning if the adopted solar metallicity is high enough to 
		potentially worry about the AGB yield grid. 
		""" 
		if self._sz[0].Z_solar > 0.018: 
			warnings.warn("""\
VICE by default implements yields from AGB stars on a grid of metallicities \
extending up to Z = 0.02. We recommend avoiding the modeling of parameter \
spaces yielding significantly super-solar total metallicities.""", 
				ScienceWarning)
		else: 
			pass 

	@property
	def agb_model(self): 
		# docstring in python class 
		return self._agb_model 

	@agb_model.setter 
	def agb_model(self, value): 
		""" 
		Keyword for AGB grid to adopt [DEPRECATED] 

		Allowed Types 
		============= 
		str [case-insensitive] 

		Allowed Values 
		============== 
		"cristallo11" 
		"karakas10" 
		""" 
		if value is not None: 
			msg = """\
Setting AGB star yield model globally in a singlezone object is deprecated in \
this version of VICE (%s). Instead, modify the yield settings for each \
element individually at vice.yields.agb.settings. Functions of stellar mass \
in solar masses and metallicity by mass (respectively) are also supported. 
""" % (_VERSION_) 
			
			# Change the settings if possible, if not just move on w/simulation  
			if isinstance(value, strcomp): 
				if value.lower() in agb._grid_reader._RECOGNIZED_STUDIES_: 
					
					# Study keywords to full journal citations. 
					studies = {
						"cristallo11": "Cristallo et al. (2011), ApJ, 197, 17", 
						"karakas10": "Karakas (2010), MNRAS, 403, 1413" 
					} 

					# Let the user know their yields will be altered. 
					msg += """
All elemental yields in the current simulation will be set to the table of \
%s with linear interpolation between masses and metallicities on the grid. 
""" % (studies[value.lower()]) 
					
					self._agb_model = value.lower() 
					for i in self.elements: 
						agb.settings[i] = value.lower() 

				else: 
					pass 
			else: 
				pass 

			msg += "\nThis feature will be removed in a future release of VICE." 
			warnings.warn(msg, VisibleDeprecationWarning) 

		else: 
			self._agb_model = None 

####### DEPRECATED IN DEVELOPMENT REPO AFTER RELEASE OF VERSION 1.0.0 #######
# 		if isinstance(value, strcomp): 
# 			if value.lower() in agb._grid_reader._RECOGNIZED_STUDIES_: 
# 				if (any(map(lambda x: atomic_number[x] > 28, self.elements)) 
# 					and value.lower() == "karakas10"): 
# 					raise LookupError("""\
# The Karakas (2010), MNRAS, 403, 1413 study did not report yields for elements \
# heavier than nickel. Please modify the attribute 'elements' to exclude these \
# elements from the simulation (or adopt an alternate yield model) before 
# proceeding.""") 
# 				else: 
# 					self._agb_model = value.lower() 
# 			else: 
# 				raise ValueError("Unrecognized AGB yield model: %s" % (
# 					value)) 
# 		else: 
# 			raise TypeError("""Attribute 'agb_model' must be of type string. \
# Got: %s""" % (type(value))) 



	# ------------------------ RUN THE SIMULATION ------------------------ # 
	def run(self, output_times, capture = False, overwrite = False): 
		
		"""
		Run's the built-in timestep integration routines over the parameters 
		built into the attributes of this class. Whether or not the user sets 
		capture = True, the output files will be produced and can be read into 
		an output object at any time. 

		Signature: vice.singlezone.run(output_times, capture = False, 
			overwrite = False) 

		Parameters 
		========== 
		output_times :: array-like [elements are real numbers] 
			The times in Gyr at which VICE should record output from the 
			simulation. These need not be sorted in any way; VICE will take 
			care of that automatically. 
		capture :: bool [default :: False] 
			A boolean describing whether or not to return an output object 
			from the results of the simulation. 
		overwrite :: bool [default :: False] 
			A boolean describing whether or not to force overwrite any existing 
			files under the same name as this simulation's output files. 

		Returns 
		======= 
		out :: output [only returned if capture = True] 
			An output object produced from this simulation's output. 

		Raises 
		====== 
		TypeError :: 
			::	Any functional attribute evaluates to a non-numerical value 
				at any timestep 
		ValueError :: 
			::	Any element of output_times is negative 
			:: 	An inflow metallicity evaluates to a negative value 
		ArithmeticError :: 
			::	An inflow metallicity evaluates to NaN or inf 
			::	Any functional attribute evaluates to NaN or inf at any 
				timestep 
		UserWarning :: 
			::	Any yield settings or class attributes are callable 
				functions and the user does not have dill installed 
		ScienceWarning :: 
			::	Any element tracked by the simulation is enriched in 
				significant part by the r-process 
			::	Any element tracked by the simulation has a weakly constrained 
				solar abundance measurement 

		Notes
		=====
		Encoding functional attributes into VICE outputs requires the 
		package dill, an extension to pickle in the python standard library. 
		Without this, the outputs will not have memory of any functional 
		attributes stored in this class. It is recommended that VICE users 
		install dill if they have not already so that they can make use of this 
		feature; this can be done via 'pip install dill'. 

		When overwrite = False, and there are files under the same name as the 
		output produced, this acts as a halting function. VICE will wait for 
		the user's approval to overwrite existing files in this case. If 
		user's are running multiple simulations and need their integrations 
		not to stall, they must specify overwrite = True. 

		Example 
		======= 
		>>> import numpy as np 
		>>> sz = vice.singlezone(name = "example") 
		>>> outtimes = np.linspace(0, 10, 1001) 
		>>> sz.run(outtimes) 
		"""
		output_times = self.prep(output_times) 
		# zoneprep(self._sz, self, output_times, overwrite) 
		cdef int enrichment 
		if self.outfile_check(overwrite): 
			if not os.path.exists("%s.vice" % (self.name)): 
				os.system("mkdir %s.vice" % (self.name)) 
			else: 
				pass 

			# warn the user about r-process elements and bad solar calibrations 
			self.nsns_warning() 
			self.solar_z_warning() 

			# just do it #nike 
			self._sz[0].output_times = copy_pylist(output_times) 
			self._sz[0].n_outputs = len(output_times) 
			enrichment = _singlezone.singlezone_evolve(self._sz) 

			# save yield settings and attributes 
			self.save_yields() 
			self.save_attributes() 

		else: 
			_singlezone.singlezone_cancel(self._sz) 
			enrichment = 0 

		if enrichment: 
			raise SystemError("Internal Error") 
		elif capture: 
			return output(self.name) 
		else: 
			pass 

	def prep(self, output_times): 
		""" 
		Prepares the simulation to be ran based on the current settings. 

		Parameters 
		========== 
		output_times :: array-like 
			The array of values the user passed to run() 

		Returns 
		======= 
		times :: list 
			A copy of the array the user passed as a list, if successful 

		Raises 
		====== 
		Exceptions raised by subroutines 
		""" 
		# Make sure the output times are as they should be 
		output_times = self.output_times_check(output_times) 
		self._sz[0].ism[0].mass = self._Mg0 # reset initial gas supply 
		setup_imf(self._sz[0].ssp[0].imf, self._imf) 

		""" 
		Construct the array of times at which the simulation will evaluate, 
		and map specified functions across those times. In the case of sfr and 
		ifr mode, the factor of 1e9 converts from Msun yr^-1 to Msun Gyr^-1. 
		""" 
		evaltimes = _pyutils.range_(0, 
			output_times[-1] + 10 * self.dt, 
			self.dt
		) 
		self.setup_elements(evaltimes) 
		if evaltimes[-1] > _singlezone.SINGLEZONE_MAX_EVAL_TIME: 
			warnings.warn("""\
VICE does not support simulations of timescales longer than %g Gyr. This 
simulation may produce numerical artifacts or a segmentation fault at late 
times.""" % (_singlezone.SINGLEZONE_MAX_EVAL_TIME, ScienceWarning))  
		else: 
			pass 

		def negative_checker(arr, name): 
			""" 
			Checks for negatives in an evaluated functional attribute 
			""" 
			if any(map(lambda x: x < 0, arr)): 
				raise ArithmeticError("""Functional attribute '%s' evaluated \
to negative value for at least one timestep.""" % (name))
			else: 
				pass 

		def mapper(attr, name, allow_inf = False): 
			""" 
			Maps numerical/functional attributes across time 

			allow_inf :: whether or not to allow infinity as a value 
				Currently only the case for attribute 'tau_star' 
			""" 
			if callable(attr): 
				arr = list(map(attr, evaltimes)) 
				_pyutils.numeric_check(arr, ArithmeticError, """Functional \
attribute '%s' evaluated to non-numerical value for at least one \
timestep.""" % (name))  
			else: 
				arr = len(evaltimes) * [attr] 
			if allow_inf: 
				# only check for NaNs, allowing infs to slip through 
				if any(list(map(m.isnan, arr))): 
					raise ArithmeticError("""Functional attribute '%s' \
evaluated to NaN for at least one timestep.""" % (name)) 
				else: pass 
			else: 
				_pyutils.inf_nan_check(arr, ArithmeticError, """Functional \
attribute '%s' evaluated to inf or NaN for at least one timestep.""" % (name)) 
				negative_checker(arr, name) 
			return arr 

		# map attributes across time  
		self._sz[0].ism[0].eta = copy_pylist(mapper(self._eta, "eta"))  
		self._sz[0].ism[0].enh = copy_pylist(mapper(
			self._enhancement, "enhancement")) 
		# Allow inf for tau_star if in infall or gas mode, but not sfr mode 
		self._sz[0].ism[0].tau_star = copy_pylist(mapper(
			self._tau_star, "tau_star", 
			allow_inf = self.mode in ["ifr", "gas"]))  
		if self.mode == "gas": 
			self._sz[0].ism[0].specified = copy_pylist(mapper(
				self._func, "func"))  
		else: 
			# 1.e9 converts from Msun yr^-1 to Msun Gyr^-1 
			self._sz[0].ism[0].specified = copy_pylist(mapper(
				lambda t: 1.e9 * self._func(t), "func")) 

		# Set a custom DTD if specified 
		if callable(self._ria): 
			self.set_ria() 
		else: pass 

		self.setup_Zin(output_times[-1]) 
		return output_times 

	def output_times_check(self, output_times): 
		""" 
		Ensures that the output times have only numerical values above zero. 

		Parameters 
		========== 
		output_times :: array-like 
			The array of values the user passed. 

		Returns 
		======= 
		times :: list 
			A copy of the array the user passed as a list, if successful 

		Raises 
		====== 
		TypeError :: 
			:: output_times is not an array-like object 
			:: Non-numerical value in the passed array 
		""" 
		output_times = _pyutils.copy_array_like_object(output_times) 
		_pyutils.numeric_check(output_times, TypeError, """Non-numerical \
value detected in output times.""") 
		output_times = sorted(output_times) 

		# Some more refinement to ensure cleaner outputs 
		return self.refine_output_times(output_times) 

	def refine_output_times(self, output_times): 
		""" 
		Removes any elements of the user's specified output_times array that 
		are less than self.dt - 1.e-5 larger than the previous output time. 
		Then ensures that output_times will not be a denser array of numbers 
		than the times at which the simulation will evaluate. 
		""" 
		arr = len(output_times) * [0.] # It can only get smaller than this 

		# This forces output at time 0 
		n = 0 
		for i in range(1, len(output_times)): 
			# if difference is very near or larger than dt 
			if output_times[i] - arr[n] - self._sz[0].dt >= -1.e-5: 
				arr[n + 1] = output_times[i] 
				n += 1 
			else: 
				# If not, exclude it 
				continue 

		return arr[:(n + 1)] 

	def outfile_check(self, overwrite): 
		"""
		Determines if any of the output files exist and proceeds according to 
		the user specified overwrite preference. 

		Parameters 
		========== 
		overwrite :: bool 
			The user's overwrite specification - True to force overwrite. 

		Returns 
		======= 
		True if the simulation can proceed and run, overwriting any files 
		that may already exist. False if the user wishes to abort. 
		""" 
		if overwrite: 
			return True 
		else: 
			outfiles = ["%s.vice/%s" % (self.name, 
				i) for i in ["history.out", "mdf.out", "ccsne_yields.config", 
					"sneia_yields.config", "params.config"]] 
			if os.path.exists("%s.vice" % (self.name)): 
				""" 
				If the user didn't specify to overwrite, see if any of the 
				output files will overwrite anything. 
				""" 
				if any(map(os.path.exists, outfiles)): 
					answer = input("""At least one of the output files \
already exists. If you continue with the integration, their contents will \
be lost.\nOutput directory: %s.vice\nOverwrite? (y | n) """ % (self.name)) 
					
					# be emphatic about it 
					while answer.lower() not in ["yes", "y", "no", "n"]: 
						answer = input("Please enter either 'y' or 'n': ") 
					# Return whether or not the user said overwrite 
					return answer.lower() in ["y", "yes"] 
				else: 
					# None of the output files exist anyway 
					return True 
			else: 
				# output directory doesn't even exist yet 
				return True 


	def setup_elements(self, evaltimes): 
		""" 
		Setup each element's AGB grid, CCSNe yield grid, and SNe Ia yield 
		""" 
		for i in range(self._sz[0].n_elements): 
			self._sz[0].elements[i][0].solar = solar_z[self.elements[i]] 
			self._sz[0].elements[i][0].primordial = primordial[self.elements[i]] 
			self._sz[0].elements[i][0].agb_grid[0].entrainment = (
				self.entrainment.agb[self.elements[i]]) 
			self._sz[0].elements[i][0].ccsne_yields[0].entrainment = (
				self.entrainment.ccsne[self.elements[i]]) 
			self._sz[0].elements[i][0].sneia_yields[0].entrainment = (
				self.entrainment.sneia[self.elements[i]]) 
			
			# agbfile = agb._grid_reader.find_yield_file(self.elements[i], 
			# 	self._agb_model)
			# _io.import_agb_grid(self._sz[0].elements[i], 
			# 	agbfile.encode("latin-1")) 
			# self.setup_sneia_yield(i) 
			# self.setup_ccsne_yield(i) 
			
			self._sz[0].elements[i][0].ccsne_yields[0].yield_ = (
				copy_pylist(map_ccsne_yield(self.elements[i]))
			) 
			self._sz[0].elements[i][0].sneia_yields[0].yield_ = (
				copy_pylist(map_sneia_yield(self.elements[i]))
			) 

			if callable(agb.settings[self.elements[i]]): 
				masses = [_ssp.main_sequence_turnoff_mass(i, 
					self.postMS) for i in evaltimes] 
				metallicities = _pyutils.range_(
					_agb.AGB_Z_GRID_MIN, 
					_agb.AGB_Z_GRID_MAX, 
					_agb.AGB_Z_GRID_STEPSIZE 
				) 
				yield_grid = map_agb_yield(self.elements[i], masses) 
				self._sz[0].elements[i][0].agb_grid[0].n_m = len(masses) 
				self._sz[0].elements[i][0].agb_grid[0].n_z = len(metallicities)  
				self._sz[0].elements[i][0].agb_grid[0].grid = (
					copy_2Dpylist(yield_grid) 
				) 
				self._sz[0].elements[i][0].agb_grid[0].m = (
					copy_pylist(masses) 
				) 
				self._sz[0].elements[i][0].agb_grid[0].z = (
					copy_pylist(metallicities) 
				)
			else: 
				agbfile = agb._grid_reader.find_yield_file(self.elements[i], 
					agb.settings[self.elements[i]]) 
				_io.import_agb_grid(self._sz[0].elements[i], 
					agbfile.encode("latin-1")) 
				


# 	def setup_ccsne_yield(self, element_index): 
# 		""" 
# 		Fills the yield array for a given element based on the user's curent 
# 		setting for that particular element. 

# 		Parameters 
# 		========== 
# 		element_index :: int 
# 			The index of the element to setup the yield for. This is simply 
# 			the position of that element's symbol in self.elements. 
# 		""" 
# 		ccyield = ccsne.settings[self.elements[element_index]] 
# 		if callable(ccyield): 
# 			# each line ensures basic requirements of the yield settings 
# 			_pyutils.args(ccyield, """Yields from core-collapse supernovae, \
# when callable, must take only one numerical parameter.""") 
# 			z_arr = _pyutils.range_(_ccsne.CC_YIELD_GRID_MIN, 
# 				_ccsne.CC_YIELD_GRID_MAX, 
# 				_ccsne.CC_YIELD_STEP
# 			) 
# 			arr = list(map(ccyield, z_arr)) 
# 			_pyutils.numeric_check(arr, ArithmeticError, """Yield as a \
# function of metallicity mapped to non-numerical value.""") 
# 			_pyutils.inf_nan_check(arr, ArithmeticError, """Yield as a \
# function of metallicity mapped to NaN or inf for at least one metallicity.""") 
# 		elif isinstance(ccyield, numbers.Number): 
# 			if m.isinf(ccyield) or m.isnan(ccyield): 
# 				raise ArithmeticError("Yield cannot be inf or NaN.") 
# 			else: 
# 				arr = len(_pyutils.range_(_ccsne.CC_YIELD_GRID_MIN, 
# 					_ccsne.CC_YIELD_GRID_MAX, 
# 					_ccsne.CC_YIELD_STEP)) * [ccyield] 
# 		else: 
# 			raise TypeError("""IMF-integrated yield from core collapse \
# supernovae must be either a numerical value or a function of metallicity. \
# Got: %s""" % (type(ccyield))) 

# 		self._sz[0].elements[element_index][0].ccsne_yields[0].yield_ = (
# 			copy_pylist(arr)) 

# 	def setup_sneia_yield(self, element_index): 
# 		""" 
# 		Fills the yield array for a given element based on the user's current 
# 		setting for that particular element 

# 		Parameters 
# 		========== 
# 		element_index :: int 
# 			The index of the element to setup the yield for. This is simply 
# 			the position of that element's symbol in self.elements. 
# 		""" 
# 		iayield = sneia.settings[self.elements[element_index]] 
# 		if callable(iayield): 
# 			# each line ensures basic requirements of the yield settings 
# 			_pyutils.args(iayield, """Yields from type Ia supernovae, when \
# callable, must take only one numerical parameter.""") 
# 			z_arr = _pyutils.range_(
# 				_sneia.IA_YIELD_GRID_MIN, 
# 				_sneia.IA_YIELD_GRID_MAX, 
# 				_sneia.IA_YIELD_STEP
# 			) 
# 			arr = list(map(iayield, z_arr)) 
# 			_pyutils.numeric_check(arr, ArithmeticError, """Yield as a \
# function of metallicity mapped to non-numerical value.""") 
# 			_pyutils.inf_nan_check(arr, ArithmeticError, """Yield as a \
# function of metallicity mapped to NaN or inf for at least one metallicity.""") 
# 		elif isinstance(iayield, numbers.Number): 
# 			if m.isinf(iayield) or m.isnan(iayield): 
# 				raise ArithmeticError("Yield cannot be inf or NaN.") 
# 			else: 
# 				arr = len(_pyutils.range_(
# 					_sneia.IA_YIELD_GRID_MIN, 
# 					_sneia.IA_YIELD_GRID_MAX, 
# 					_sneia.IA_YIELD_STEP
# 				)) * [iayield] 
# 		else: 
# 			raise TypeError("""IMF-integrated yield from type Ia supernovae \
# must be either a numerical value or a function of metallicity. Got: %s""" % (
# 				type(iayield))) 

# 		self._sz[0].elements[element_index][0].sneia_yields[0].yield_ = (
# 			copy_pylist(arr)) 

	def set_ria(self): 
		""" 
		Maps a custom SNe Ia DTD across the evalutation times of the 
		simulation. 

		Raises 
		====== 
		ArithmeticError :: 
			:: custom function evaluates to negative, inf, or NaN 
		""" 
		assert callable(self._ria), "self._ria not callable" 

		# SNe Ia DTDs always evaluated up to some maximum time 
		times = _pyutils.range_(0, _sneia.RIA_MAX_EVAL_TIME, self.dt) 
		ria = len(times) * [0.] 
		for i in range(len(ria)): 
			# Take into account the intrinsic delay
			if times[i] >= self.delay: 
				ria[i] = self._ria(times[i]) 
			else: 
				continue 

			if (m.isnan(ria[i]) or m.isinf(ria[i]) or ria[i] < 0): 
				raise ArithmeticError("""Custom SNe Ia DTD evaluated to \
negative, NaN, or inf for at least one timestep.""") 
			else: 
				continue 

		"""
		setup_RIa in src/sneia.c will do the normalization, no need to worry 
		about that here. 
		"""
		for i in range(self._sz[0].n_elements): 
			self._sz[0].elements[i][0].sneia_yields[0].RIa = (
				copy_pylist(ria)) 

	def setup_Zin(self, endtime): 
		""" 
		Fills infall metallicity information for each element. 

		Parameters 
		========== 
		endtime :: real number 
			The ending time of the simulation 
		""" 
		assert endtime > 0, "Endtime < 0" 
		# The times at which the simulation will evaluate
		evaltimes = _pyutils.range_(0, endtime + 10 * self.dt, self.dt) 

		# maps a function of metallicity across time 
		def zin_mapper(func): 
			assert callable(func) 
			# sanity checks on what it evaluates to 
			_pyutils.args(func, """Infall metallicity, when callable, must \
accept only one numerical parameter.""") 
			arr = list(map(func, evaltimes)) 
			_pyutils.numeric_check(arr, ArithmeticError, """Infall \ 
metallicity evaluated to non-numerical value for at least one timestep.""") 
			_pyutils.inf_nan_check(arr, ArithmeticError, """Infall \
metallicity evaluated to NaN or inf for at least one timestep.""") 
			if any(map(lambda x: x < 0, arr)): 
				raise ArithmeticError("""Infall metallicity evaluated to \
negative value for at least one timestep.""") 
			else: 
				return arr 

		if isinstance(self._zin, numbers.Number): 
			# float for all elements; no need for zin_mapper 
			for i in range(self._sz[0].n_elements): 
				self._sz[0].elements[i][0].Zin = copy_pylist(
					len(evaltimes) * [self._zin]) 

		elif isinstance(self._zin, evolutionary_settings): 
			# Separate specification for each element 
			for i in range(self._sz[0].n_elements): 

				# number for this element 
				if isinstance(self._zin[self.elements[i]], numbers.Number): 
					self._sz[0].elements[i][0].Zin = copy_pylist(
						len(evaltimes) * [self._zin[self.elements[i]]]) 

				# function for this element 
				elif callable(self._zin[self.elements[i]]): 
					self._sz[0].elements[i][0].Zin = copy_pylist(
						zin_mapper(self._zin[self.elements[i]])) 

				else: 
					# failsafe 
					raise SystemError("Internal Error") 

		elif callable(self._zin): 
			# function for all elements; call zin_mapper 
			arr = zin_mapper(self._zin) 
			# Give each element a copy 
			for i in range(self._sz[0].n_elements): 
				self._sz[0].elements[i][0].Zin = copy_pylist(arr) 

		else: 
			# failesafe 
			raise SystemError("Internal Error") 

	def save_yields(self): 
		""" 
		Writes the .config yield files to the output directory. 
		""" 
		# Take a snapshot of the current yield settings 
		ccsne_yields = self._sz[0].n_elements * [None] 
		sneia_yields = self._sz[0].n_elements * [None] 
		agb_yields = self._sz[0].n_elements * [None] 
		for i in range(self._sz[0].n_elements): 
			ccsne_yields[i] = ccsne.settings[self.elements[i]] 
			sneia_yields[i] = sneia.settings[self.elements[i]] 
			agb_yields[i] = agb.settings[self.elements[i]] 

		# Turn them back into dictionaries 
		ccsne_yields = dict(zip(self.elements, ccsne_yields)) 
		sneia_yields = dict(zip(self.elements, sneia_yields)) 
		agb_yields = dict(zip(self.elements, agb_yields)) 

		def forget_functionals(settings, name): 
			""" 
			Modifies local copy of functional yield to None in the case that 
			dill is not installed. 

			Parameters 
			========== 
			settings :: dict 
				The local copy of the current settings 
			name :: str 
				The name of the enrichment channel 

			Returns 
			======= 
			A (modified, if necessary) copy of the settings 

			Raises 
			====== 
			UserWarning :: 
				::	Dill is not installed and there's a functional yield 
			""" 
			encoded = tuple(filter(lambda x: callable(settings[x.lower()]), 
				self.elements)) 
			if len(encoded) > 0: 
				if "dill" not in sys.modules: 
					msg = """\
Encoding functional yields from %s along with VICE outputs requires the \
package dill (installable via pip). Yields for the following elements will \
not be saved: """ % (name) 
					for i in encoded: 
						msg += "%s " % (i) 
						settings[i.lower()] = None 
					warnings.warn(msg, UserWarning) 
				else: 
					pass 
			else: 
				pass 
			return settings 

		ccsne_yields = forget_functionals(ccsne_yields, "CCSNe") 
		sneia_yields = forget_functionals(sneia_yields, "SNe Ia") 
		agb_yields = forget_functionals(agb_yields, "AGB stars") 

		# pickle the dataframes 
		pickle.dump(ccsne_yields, open("%s.vice/ccsne_yields.config" % (
			self.name), "wb")) 
		pickle.dump(sneia_yields, open("%s.vice/sneia_yields.config" % (
			self.name), "wb")) 
		pickle.dump(agb_yields, open("%s.vice/agb_yields.config" % (
			self.name), "wb")) 

	def save_attributes(self): 
		""" 
		Saves the .config file to the output directory containing all of the 
		attributes. 
		""" 

		"""
		Passing this dictionary as **kwargs to self.__init__ initializes the 
		exact same singlezone object. 
		""" 
		params = {
			"agb_model":			self.agb_model,  
			"bins": 				self.bins, 
			"delay": 				self.delay, 
			"dt": 					self.dt, 
			"RIa": 					self.RIa, 
			"elements": 			self.elements, 
			"enhancement": 			self.enhancement, 
			"entrainment.agb": 		self.entrainment.agb.todict(), 
			"entrainment.ccsne": 	self.entrainment.ccsne.todict(), 
			"entrainment.sneia": 	self.entrainment.sneia.todict(), 
			"eta": 					self.eta, 
			"func": 				self.func, 
			"IMF": 					self.IMF, 
			"m_lower": 				self.m_lower, 
			"m_upper": 				self.m_upper, 
			"Mg0":					self.Mg0, 
			"MgSchmidt": 			self.MgSchmidt, 
			"mode": 				self.mode, 
			"name": 				self.name, 
			"recycling": 			self.recycling, 
			"schmidt": 				self.schmidt, 
			"schmidt_index": 		self.schmidt_index, 
			"smoothing": 			self.smoothing, 
			"tau_ia": 				self.tau_ia, 
			"tau_star": 			self.tau_star, 
			"verbose": 				self.verbose, 
			"Z_solar": 				self.Z_solar, 
			"Zin": 					self.Zin
		} 
 
		if "dill" not in sys.modules: 
			# user doesn't have dill. functional attributes switch to None 
			functional = [] 
			for i in params.keys(): 
				""" 
				Check for callable functions in evolutionary_settings 
				attributes 
				""" 
				if isinstance(params[i], evolutionary_settings): 
					# can't pickle cdef objects, so pickle the dictionaries 
					params[i] = params[i].todict() 
					for j in params[i].keys(): 
						if callable(params[i][j]): 
							params[i][j] = None 
							functional.append("%s(%s)" % (i, j)) 
						else: 
							continue 
				elif callable(params[i]): 
					params[i] = None 
					functional.append(i) 
				else: 
					continue 

			if len(functional) > 0: 
				message = """\
Saving functional attributes within VICE outputs requires dill (installable \
via pip). The following functional attributes will not be saved with this \
output: """ 
				for i in functional: 
					message += "%s " % (i) 
				warnings.warn(message, UserWarning) 
			else: 
				pass 
		else: 
			pass 

		pickle.dump(params, open("%s.vice/params.config" % (self.name), "wb")) 

	def nsns_warning(self): 
		""" 
		Determines which, if any, or the tracked elements are enriched via the 
		r-process. In this case, VICE raises a ScienceWarning that these 
		elements will be under-abundant in the simulation. 
		""" 
		rprocess = list(filter(lambda x: "NSNS" in sources[x], self.elements)) 

		# Anything that survived the filter comes from the r-process 
		if len(rprocess) > 0: 
			message = """The following elements tracked by this simulation \
believed to be enriched by the r-process: """ 
			for i in rprocess: 
				message += "%s " % (i) 
			message += """\n\
In its current version, VICE is not designed to model enrichment via the \
r-process. These elements will likely be under-abundant in the output.""" 
			warnings.warn(message, ScienceWarning) 
		else: 
			pass 

	def solar_z_warning(self): 
		""" 
		Determines if VICE is about to simulate the enrichment of an element 
		whose solar Z calibration off of Asplund et al. (2009) is not well 
		understood. Raises a ScienceWarning that the trnds should be 
		interpreted as having arbitrary normalization in these cases. 
		""" 
		poorly_calibrated = tuple(["as", "se", "br", "cd", "sb", "te", "i", 
			"cs", "ta", "re", "pt", "hg", "bi"]) 
		test = list(filter(lambda x: x.lower() in poorly_calibrated, 
			self.elements)) 
		if len(test) > 0: 
			message = """\
The following elements do not have a well known solar abundance: """ 
			for i in test: 
				message += "%s " % (i) 
			message += """\n\
For this reason, the [X/H] abundances relative to the sun and all [X/Y] 
abundance ratios involving these elements should be interpreted as having an \
arbitrary normalization.""" 
			warnings.warn(message, ScienceWarning) 
		else: 
			pass 

