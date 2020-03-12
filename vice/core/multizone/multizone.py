
from __future__ import absolute_import 
from ..._globals import _VERSION_ERROR_ 
from ._multizone import c_multizone 
from ..outputs._output_utils import _check_singlezone_output 
from ..outputs._output_utils import _is_multizone 
from ..outputs._output_utils import _get_name 
from ..outputs import multioutput 
from ..outputs import output 
from .. import pickles 
import warnings 
import numbers 
import sys 
if sys.version_info[:2] == (2, 7): 
	strcomp = basestring 
elif sys.version_info[:2] >= (3, 5): 
	strcomp = str 
else: 
	_VERSION_ERROR_() 

""" 
NOTES 
===== 
cdef class objects do not transfer the docstrings of class attributes to the 
compiled output, leaving out the internal documentation. For this reason, 
wrapping of the multizone object has two layers -> a python class and a 
C class. In the python class, there is only one attribute: the C version of 
the wrapper. The docstrings are written here, and each function/setter 
only calls the C version of the wrapper. While this is a more complicated 
wrapper, it preserves the internal documentation. In order to maximize 
readability, the setter functions of the C version of the wrapper have brief 
notes on the physical interpretation of each attribute as well as the allowed 
types and values. 
""" 

class multizone(object): 

	""" 
	Runs simulations of chemical enrichment under the multi-zone approximation 
	for user-specified parameters. 

	Signature vice.multizone.__init__(name = "multizonemodel", 
		n_zones = 10, 
		n_tracers = 1, 
		verbose = False) 

	Attributes 
	========== 
	name :: str [default :: "multizonemodel"] 
		The name of the simulation 
	n_zones :: int [default :: 10] 
		The number of zones in the simulation 
	n_tracers :: int [default :: 1] 
		The number of tracer particles per zone per timestep 
	verbose :: bool [default :: False] 
		Whether or not to print the time to the console as the simulation runs 
	migration :: object [default :: zeroes] 
		The migration matrices specifying how gas and stellar tracer particles 
		should be moved between zones at each timestep. 

	Functions 
	========= 
	run :: 
		Run the simulation 

	See also 	[https://github.com/giganano/VICE/tree/master/docs] 
	======== 
	VICE's science documentation 
	""" 

	def __new__(cls, n_zones = 10, **kwargs): 
		""" 
		__new__ is overridden such that a singlezone object is returned 
		when n_zones = 1. 
		""" 
		if isinstance(n_zones, numbers.Number): 
			if n_zones > 0: 
				if n_zones % 1 == 0: 
					n_zones = int(n_zones) 
					if n_zones == 1: 
						return singlezone() 
					else: 
						return super(multizone, cls).__new__(cls) 
				else: 
					raise ValueError("""Attribute 'n_zones' must be of type \
int. Got: %g""" % (n_zones)) 
			else: 
				raise ValueError("Attribute 'n_zones' must be non-negative.") 
		else: 
			raise TypeError("""Attribute 'n_zones' must be of type int. \
Got: %s""" % (type(n_zones))) 

	def __init__(self, n_zones = 10, **kwargs): 
		""" 
		All attributes can be specified as a keyword argument. 

		Notes 
		===== 
		When n_zones = 1, a singlezone object is initialized instead 
		""" 
		self.__c_version = c_multizone(n_zones = int(n_zones), **kwargs) 

	def __repr__(self): 
		""" 
		Prints in the format: vice.singlezone{ 
			attr1 -----------> value 
			attribute2 ------> value 
		}
		""" 
		attrs = {
			"name": 			self.name, 
			"n_zones": 			self.n_zones, 
			"n_tracers": 		self.n_tracers, 
			"verbose": 			self.verbose, 
			"simple": 			self.simple, 
			"zones": 			[self.zones[i].name for i in range(
									self.n_zones)], 
			"migration": 		self.migration 
		} 

		rep = "vice.multizone{\n" 
		for i in attrs.keys(): 
			rep += "    %s " % (i) 
			for j in range(15 - len(i)): 
				rep += '-' 
			rep += "> %s\n" % (str(attrs[i])) 
		rep += '}' 
		return rep 

	def __str__(self): 
		return self.__repr__() 

	def __enter__(self): 
		""" 
		Opens a with statement 
		""" 
		return self 

	def __exit__(self, exc_type, exc_value, exc_tb): 
		""" 
		Raises all exceptions inside with statements 
		""" 
		return exc_value is None 

	@classmethod  
	def from_output(cls, arg): 
		""" 
		Obtain an instance of the vice.multizone class given either the path 
		to an output or a multioutput itself. 

		Signature: vice.multizone.from_output(arg) 

		Parameters 
		========== 
		arg :: str or vice.multioutput 
			The full or relative path to the multioutput directory. 
			Alternatively, the multioutput object. 

		Returns 
		======= 
		mz :: vice.multizone 
			A multizone object with the same parameters as the one which 
			produced the multioutput. 

		Raises 
		====== 
		TypeError :: 
			::	arg is neither a multioutput object nor a string 
		IOError :: 
			::	output is not found, or is missing files 

		Notes 
		===== 
		If arg corresponds to either a singlezone output or an output object, 
		a singlezone object is returned. 

		See Also 
		======== 
		vice.mirror 

		Added: 1.1.0 
		""" 

		""" 
		Developer's Notes 
		================= 
		While this function serves as the reader, the writer is the 
		vice.core.multizone._multizone.c_multizone.pickle function, 
		implemented in cython. Any changes to this function should be reflected 
		there. 
		""" 
		if isinstance(arg, multioutput): 
			# recursion to the algorithm which does it from the path 
			return cls.from_output(arg.name) 
		elif isinstance(arg, output): 
			""" 
			Return the corresponding singlezone object. 
			These import statements are here to prevent ImportErrors caused by 
			nested recursive imports. 
			""" 
			from ..singlezone import singlezone 
			return singlezone.from_output(arg) 
		elif isinstance(arg, strcomp): 
			dirname = _get_name(arg) 
			if not _is_multizone(dirname): 
				from ..singlezone import singlezone 
				return singlezone.from_output(dirname) 
		else: 
			raise TypeError("""Must be either a string or an output object. \
Got: %s""" % (type(arg))) 

		from ..singlezone import singlezone 
		attrs = pickles.jar.open("%s/attributes" % (dirname)) 
		mz = cls(n_zones = attrs["n_zones"]) 
		mz.name = attrs["name"] 
		mz.n_tracers = attrs["n_tracers"] 
		mz.simple = attrs["simple"] 
		mz.verbose = attrs["verbose"] 
		for i in range(mz.n_zones): 
			mz.zones[i] = singlezone.from_output("%s/%s.vice" % (dirname, 
				attrs["zones"][i])) 
			mz.zones[i].name = attrs["zones"][i] 
		
		stars = pickles.jar.open("%s/migration" % (dirname))["stars"] 
		if stars is None: 
			warnings.warn("""\
Attribute not encoded with output: migration.stars. Assuming default value, \
which may not reflect the value of this attribute at the time the simulation \
was ran.""", UserWarning) 
		else: 
			mz.migration.stars = stars 

		for i in range(mz.n_zones): 
			attrs = pickles.jar.open("%s/migration/gas%d" % (dirname, i)) 
			for j in range(mz.n_zones): 
				if attrs[str(j)] is None: 
					warnings.warn("""\
Attribute not encoded with output: migration.gas[%d][%d]. Assuming default \
value, which may not reflect the value of this attribute at the time the \
simulation was ran.""" % (i, j), UserWarning) 
				else: 
					mz.migration.gas[i][j] = attrs[str(j)]  

		return mz 


	@property 
	def name(self): 
		""" 
		Type :: str 
		Default :: "multizonemodel" 

		The name of the simulation. The output will be stored in a directory 
		under this name with the extension ".vice". This can also be of the 
		form /path/to/directory/name and the output will be stored there. 

		Notes 
		===== 
		The user need not interact with any of the output files; the output 
		object is designed to read in all of the results automatically. 

		By forcing a ".vice" extension on the output file, users can run 
		'<command> *.vice' in a linux terminal to run commands over all VICE 
		outputs in a given directory. 

		See Also 
		======== 
		vice.singlezone 
		vice.singlezone.name 
		""" 
		return self.__c_version.name 

	@name.setter 
	def name(self, value): 
		self.__c_version.name = value 

	@property 
	def zones(self): 
		""" 
		Type :: array-like 
		
		An array-like object whose elements are the singlezone objects 
		corresponding to each individual zone in the simulation. Since the 
		elements of this property are all singlezone objects, their attributes 
		and output may all be manipulated as such. 

		Notes 
		===== 
		The output associated with each zone will be stored inside the output 
		directory from this class. For example, for a multizone object whose 
		name is "multizonemodel" with a zone named "onezonemodel", the output 
		will be stored in the path: 

		multizonemodel.vice/onezonemodel.vice 

		See Also 
		======== 
		vice.singlezone 
		vice.singlezone.name 
		""" 
		return self.__c_version.zones 

	@property 
	def migration(self): 
		""" 
		Type :: object 

		The migration specifications of the multizone model. For a simulation 
		with N zones, the migration matrix is NxN, where the ij'th element 
		represents the likelihood that either gas or stars migrate OUT OF the 
		i'th zone and INTO the j'th zone. 

		Attributes 
		========== 
		stars :: object 
			The migration matrix for tracer particles of stellar populations 
		gas :: object 
			The migration matrix for interstellar gas 

		Notes 
		===== 
		By default, both migration matrices have elements that default to 
		zero, meaning that by default this object runs N singlezone simulations 
		with stars and gas that never migrate between zones. It is up to the 
		user to specify each individual likelihood. 
		""" 
		return self.__c_version.migration 

	@property 
	def n_zones(self): 
		""" 
		Type :: int 
		Default :: 10 

		The number of zones in the simulation. 

		Notes 
		===== 
		Users may only manipulate the value of thie object upon initialization 
		of the multizone object. In order to change the number of zones in a 
		multizone simulation, a new multizone object must be initialized. 
		""" 
		return self.__c_version.n_zones 

	@property 
	def n_tracers(self): 
		""" 
		Type :: int 
		Default :: 1 

		The number of tracer particles per zone per timestep. These tracer 
		particles represent the stellar populations that form in each zone, 
		and migrate between zones according to the user-specified migration 
		matrix. 
		""" 
		return self.__c_version.n_tracers 

	@n_tracers.setter 
	def n_tracers(self, value): 
		self.__c_version.n_tracers = value 

	@property 
	def verbose(self): 
		""" 
		Type :: bool 
		Default :: False 

		If True, the time in Gyr will print to the console as the simulation 
		evolves. 
		""" 
		return self.__c_version.verbose 

	@verbose.setter 
	def verbose(self, value): 
		self.__c_version.verbose = value 

	@property 
	def simple(self): 
		""" 
		Type :: bool 
		Default :: True 

		If False, the tracer particles' zone numbers at each intermediate 
		timestep will be taken into account. Otherwise, each zone will 
		evolve independently of one another, and the metallicity distribution 
		functions will be computed from the final positions of each tracer 
		particle. 
		""" 
		return self.__c_version.simple 

	@simple.setter 
	def simple(self, value): 
		self.__c_version.simple = value 

	def run(self, output_times, capture = False, overwrite = False): 
		""" 
		Run's the built-in timestep integration routines over the parameters 
		built into the attributes of this class as well as the individual 
		zones associated with it. Whether or not the user sets capture = True, 
		the output files will be produced and can be read into an output 
		object at any time. 

		Signature: vice.multizone.run(output_times, capture = False, 
			overwrite = False) 

		Parameters 
		========== 
		output_times :: array-like [elements are real numbers] 
			The time in Gyr at which VICE should record output from the 
			simulation. These need not be sorted in any way; VICE will take 
			care of that automatically. 
		capture :: bool [default :: False] 
			A boolean describing whether or not to return an output object 
			from the results of the simulation. 
		overwrite :: bool [default :: False] 
			A boolean describing whether or not to force overwrite any 
			existing files under the same name as this simulation. 

		Returns 
		======= 
		out :: vice.dataframe [only returned if capture = True] 
			A VICE dataframe relating each zone to its associated output 
			object. 

		Raises 
		====== 
		RuntimeError :: 
			::	A migration matrix cannot be setup properly according to the 
				user's current specifications 
			::	Any of the zones associated with this object have duplicate 
				names 
			:: 	The timestep size is not uniform across each zone 
		ScienceWarning :: 
			::	Any of the attributes 'IMF', 'recycling', 'delay', 'RIa', 
				'schmidt', 'schmidt_index', 'MgSchmidt', 'm_upper', 'm_lower', 
				'Z_solar', and 'agb_model' aren't uniform across all zones. 
				Realistically these attributes would be, but this is not 
				required for the simulation to run properly. 
		Other exceptions raised by vice.singlezone.run 

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
		>>> mz = vice.multizone(name = "example") 
		>>> outtimes = np.linspace(0, 10, 1001) 
		>>> mz.run(outtimes) 
		""" 
		return self.__c_version.run(output_times, capture = capture, 
			overwrite = overwrite) 

