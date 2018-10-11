import glob
import sys
import numpy as np

from PIL import Image

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

print(model.summary())

def ExtractImageFeature(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	output = model.predict(x)
	print(output.shape)
	return output

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

			if idx % 50 == 0:
				SaveDict(result)

	print("Script done!")