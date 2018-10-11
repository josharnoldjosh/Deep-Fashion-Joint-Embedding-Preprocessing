from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import glob
import numpy as np
from PIL import Image
import sys
import deepdish as dd

# Function to extract image features
model = VGG16(weights='imagenet', include_top=False)
def ExtractImageFeature(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	img_data = image.img_to_array(img)
	img_data = np.expand_dims(img_data, axis=0)
	img_data = preprocess_input(img_data)
	vgg16_feature = model.predict(img_data)
	return np.array(vgg16_feature).flatten()

def SaveDict(data):
	import pickle	
	with open('data.pkl', 'wb') as f:	    
		print("saving...")
		pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
		print("saved!")

if __name__ == "__main__":

	# Where we will search for images
	path_to_images = sys.argv[1]

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

			if idx % 100 == 0:
				SaveDict(result)

	print("Script done!")