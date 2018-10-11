import json
import pickle
import random
import numpy as np

# Load dictionary
img_vecs = None
with open('image_feature_dictionary.pkl', 'rb') as handle:
	img_vecs = pickle.load(handle)	

def seperate_uniform(data):
	captions = ""
	vecs = []

	for i in data:
		captions += i[0] + "\n"
		vecs.append(i[1])

	return captions.rstrip(), np.array(vecs)

def save_data(data, split=0.1):

	# Shuffle data
	random.shuffle(data)

	# Split data to test & train
	split_index = int(len(data)*split)
	test = data[:split_index]
	train = data[split_index:]

	test_cap, test_vec = seperate_uniform(test)
	train_cap, train_vec = seperate_uniform(train)

	# Save 
	np.save('deepfashion/deepfashion_dev_ims.npy', test_vec)	 
	with open("deepfashion/deepfashion_dev_caps.txt", "w") as handle:
		handle.write(test_cap)

	np.save('deepfashion/deepfashion_train_ims.npy', train_vec)
	with open("deepfashion/deepfashion_train_caps.txt", "w") as handle:
		handle.write(train_cap)

def get_batch(item_id, caption):
	# result
	result = []

	# Extract image features from dictionary
	image_features = img_vecs[item_id]
	for vec in image_features:
		result.append((caption, vec))

	return result

def extract_item_info(item):
	item_id = item["item"] # Item id, e.g : id_00000001
	captions = item["description"] # Caption
	if len(captions) > 0:
		return True, item_id, captions[0] # Valid, item id, long caption
	return False, item_id, "", # Invalid, item id, empty caption

def main():
	with open("list_description_inshop.json") as f:

		# Load data
		data = json.load(f)

		# result
		result = []

		# Loop through each item in the data 
		for item in data:

			# Extract item info
			is_valid, item_id, caption = extract_item_info(item)

			# If valid
			if is_valid:
				result += get_batch(item_id, caption)

		# Save our result.
		save_data(result)

if __name__ == '__main__':
	main()
	print("Script done.")