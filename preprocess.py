import json
import pickle
import numpy as np

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
		self.train = ([], [])
		self.test = ([], [])
		self.sanity()

	def addTest(self, caption, vecs):
		for idx, vec in enumerate(vecs):
			if len(vec) > 0:
				self.test[0].append(caption)
				self.test[1].append(vec[0]) 
		self.sanity()

	def addTrain(self, caption, vecs):
		for idx, vec in enumerate(vecs):
			if len(vec) > 0:
				self.train[0].append(caption)
				self.train[1].append(vec[0])
		self.sanity()

	def sanity(self):
		if len(self.train[0]) != len(self.train[1]):
			print("Train is not same length!")
		if len(self.test[0]) != len(self.test[1]):
			print("Test is not same length!")

	def saveArray(self, data, name="train"):		
		# Captions		
		captions = ""
		for cap in data[0]:
			captions += cap + "\n"					
		open("deepfashion/deepfashion_"+name+"_caps.txt", "w").write(captions.rstrip())

		# Vectors 
		vecs = np.array(data[1])		
		np.save('deepfashion/deepfashion_'+name+'_ims.npy', vecs)

		# Final sanity check
		print("Difference (should be zero): ", (len(captions.split("\n"))-1) - len(vecs))
		return

	def saveOutput(self):
		self.saveArray(self.train, "train")
		self.saveArray(self.test, "dev")
		return

# Load data, output, and image features
data = Data()
output = Output()
image_vecs = pickle.load(open("image_feature_dictionary.pkl", "rb"))

# Test - train split
test_size = 1600 # = 0.2 * 8000 ~ 20% of data

for idx, caption in data:
	if caption.strip() != "":
		if data.counter+1 > test_size:
			output.addTrain(caption.replace("\n", ""), image_vecs[idx])
		else:
			output.addTest(caption.replace("\n", ""), image_vecs[idx])

# Save our output
output.saveOutput()

# Output final lengths of data
print("Test len, train len:", len(output.test[0]), len(output.train[0]))

# Sanity check
output.sanity()

# Done.
print("Script done.")
