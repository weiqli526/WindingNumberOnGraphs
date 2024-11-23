import os
# print(os.getcwd())

from PIL import Image
import numpy as np
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import torch
import open_clip

samples_num = 110

def generate_digit():
    image_dir = '../training_images'
    image_paths = [
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))
    ]

    # images = []
    # for image_file in image_paths:
    #     with Image.open(image_file) as image:
    #         image = image.resize((28, 28))
    #         images.append(np.array(image))

    # X = np.array(images)[:120]
    # X = X.reshape(-1, 28*28)

    model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w',
                                                                 pretrained='laion2b_s13b_b82k_augreg')
    embeddings = []

    image_paths = image_paths[:samples_num]
    
    for image_file in image_paths:
        print(image_file)
        with Image.open(image_file) as image:
            image = image.resize((28, 28))
            preprocessed_image = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                image_embedding = model.encode_image(preprocessed_image)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            embeddings.append(image_embedding.squeeze().numpy())

    X = np.array(embeddings)

    print(X.shape)

    Y = [int(os.path.basename(image_paths[i]).split('_')[1].split('.')[0]) for i in range(len(image_paths))][:samples_num]
    Y = np.array(Y)

    print(Y)


    X_reordered = X
    y_reordered = Y

    dir_graph = f"graphs/{y_reordered.shape[0]}/"
    os.makedirs(dir_graph, exist_ok=True)

    np.save(f'{dir_graph}/positions.npy', X_reordered)
    np.save(f'{dir_graph}/labels.npy', y_reordered)


def generate_naive_embeddings():
    image_dir = '../training_images'
    image_paths = [
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))
    ]

    images = []
    for image_file in image_paths:
        with Image.open(image_file) as image:
            image = image.resize((25, 25))
            images.append(np.array(image))

    X = np.array(images)[:samples_num]
    X = X.reshape(-1, 25*25)

    Y = [int(os.path.basename(image_paths[i]).split('_')[1].split('.')[0]) for i in range(len(image_paths))][:samples_num]
    Y = np.array(Y)

    dir_graph = f"graphs/{X.shape[0]}/"
    os.makedirs(dir_graph, exist_ok=True)

    np.save(f'{dir_graph}/positions.npy', X)
    np.save(f'{dir_graph}/labels.npy', Y)


def modify_data():
    dir_graph = f"graphs/{samples_num}/"
    X = np.load(f'{dir_graph}/positions.npy')
    Y = np.load(f'{dir_graph}/labels.npy')

    centers = np.array([X[Y == c].mean(axis=0) for c in np.unique(Y)])

    distances = pairwise_distances(X, centers)

    nearest_dist_to_self_class = np.min(
        np.where(np.eye(len(centers))[Y.astype(int)] == 1, distances, np.inf),
        axis=1
    )

    sorted_indices = np.argsort(nearest_dist_to_self_class)[::-1]

    X_reordered = X[sorted_indices]
    y_reordered = Y[sorted_indices]

    print("Reordered shapes:")
    print(X_reordered.shape)
    print(y_reordered.shape)


    np.save(f'{dir_graph}/positions.npy', X_reordered)
    np.save(f'{dir_graph}/labels.npy', y_reordered)


# generate_digit()
generate_naive_embeddings()
modify_data()