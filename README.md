# Deep-Fashion-Joint-Embedding-Preprocessing

## Introduction

Code for preprocessing the deep fashion dataset as an input to my joint embedding model.

[Joint embedding model](https://github.com/josharnoldjosh/Image-Caption-Joint-Embedding)

[Deep fashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

## Steps

1. Download the deep fashion dataset from link above.

2. Drag `extract_image_features.py` and `img_2_vec.py` so it is **next** to the deep fashion folder you downloaded.

3. Rename the deep fashion folder to `In-shop Clothes Retrieval Benchmark`. Inside of the folder should be `Anno`, `Img` folders, etc.

4. Run `extract_image_features.py`

5. Next, drag the output `image_feature_dictionary.pkl` and `preprocess.py` into the `In-shop Clothes Retrieval Benchmark/Anno` and then run `preprocess.py`.

6. You should have an output folder `deepfashion/` containing the input for the model. Just drop the `deepfashion/` into the `data/` folder.
