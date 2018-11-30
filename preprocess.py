import json
import pickle
import numpy as np
import clean

class Data:
	def __init__(self):
		self.data = json.load(open("list_description_inshop.json"))
		self.counter = -1

	def __next__(self):
		if len(self.data) > self.counter+1:
			self.counter += 1
			return self.data[self.counter]["item"], (self.data[self.counter]["description"][0] if len(self.data[self.counter]["description"]) > 0 else "")
		else:
			self.counter = 0
			raise StopIteration

	def __iter__(self):
		return self

class Output:
	def __init__(self):
		self.data = ([], [])
		self.sanity()

	def add(self, caption, vecs):
		for idx, vec in enumerate(vecs):
			if len(vec) > 0:
				self.data[0].append(caption)
				self.data[1].append(vec[0]) 
		self.sanity()

	def sanity(self):
		if len(self.data[0]) != len(self.data[1]):
			print("Data is not same length!")

	def save(self, name=""):		
		# Captions		
		captions = ""
		for cap in self.data[0]:
			captions += cap + "\n"					
		open("deepfashion/deepfashion_caps.txt", "w").write(captions.rstrip())

		# Vectors 
		vecs = np.array(self.data[1])		
		np.save('deepfashion/deepfashion_ims.npy', vecs)

		# Final sanity check
		print("Difference (should be zero): ", (len(captions.split("\n"))-1) - len(vecs))
		return

# Load data, output, and image features
data = Data()
output = Output()
image_vecs = pickle.load(open("image_feature_dictionary.pkl", "rb"))

for idx, caption in data:			
	output.add(clean.caption(caption), image_vecs[idx])	

# Save our output
output.save()

# Sanity check
output.sanity()

# Done.
print("Script done.")
