import glob
import sys
import numpy as np
from img_to_vec import Img2Vec
from PIL import Image

img2vec = Img2Vec(cuda=True)

def ExtractImageFeature(path):
	img = Image.open(path)
	return img2vec.get_vec(img)	

def SaveDict(data):
	import pickle	
	with open('image_feature_dictionary.pkl', 'wb') as f:	    
		print("saving...")
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		print("saved!")

if __name__ == "__main__":

	# Where we will search for images
	path_to_images = "In-shop Clothes Retrieval Benchmark/" #sys.argv[1]

	# Dictionary to store results
	result = {}

	idx = 0

	# Search all sub directories
	for filename in glob.iglob(path_to_images+"**/*/", recursive=True):	

		idx += 1

		# Find images in directory
		files = (glob.glob(filename+"*.jpg") + glob.glob(filename+"*.png"))

		# Create array of extracted image features
		vectors = [ExtractImageFeature(file_path) for file_path in files]

		# Get "image id" of folder
		image_id = filename.rsplit('/')[-2]

		# Check it is a valid ID
		if "id_" in image_id:
			# If so, let's map this image id to image vectors.
			print(image_id)		
			result[image_id] = vectors	

			if idx % 50 == 0:
				SaveDict(result)

	# Save final dictionary
	SaveDict(result)

	print("Script done!")