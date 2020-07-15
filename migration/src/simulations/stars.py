
from vice.toolkit import hydrodisk 
from .config import config 


class diskmigration(hydrodisk.hydrodiskstars): 

	def __init__(self, radbins, mode = "linear", filename = "stars.out"): 
		super().__init__(radbins, mode = mode) 
		if isinstance(filename, str): 
			self._file = open(filename, 'w') 
			self._file.write("# zone_origin\ttime_origin\tzfinal\n") 
		else: 
			raise TypeError("Filename must be a string. Got: %s" % (
				type(filename))) 
		self.write = False 

	def __call__(self, zone, tform, time): 
		if tform == time: 
			super().__call__(zone, tform, time) # reset analog star particle 
			if self.write: 
				if self.analog_index == -1: 
					finalz = 100 
				else: 
					finalz = self.analog_data["zfinal"][self.analog_index] 
				self._file.write("%d\t%.2f\t%.2f\n" % (zone, tform, finalz)) 
			else: pass 
			return zone 
		else: 
			return super().__call__(zone, tform, time) 

	@property 
	def write(self): 
		r""" 
		Type : bool 

		Whether or not to write out to the extra star particle data output 
		file. For internal use by the vice.multizone object only. 
		""" 
		return self._write 

	@write.setter 
	def write(self, value): 
		if isinstance(value, bool): 
			self._write = value 
		else: 
			raise TypeError("Must be a boolean. Got: %s" % (type(value))) 
