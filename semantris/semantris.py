#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pyautogui
import annoy
import random
import pickle
import os
import pyperclip
import keras_ocr
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub



def screenshot(bbox=None): #bbox format : left - top - width - height
    return pyautogui.screenshot(region=bbox)

FIGSIZE = (6,6)
def plot_img(image, figsize=FIGSIZE,cmap=None,image_name = 'image.png'):
    plt.figure(figsize=figsize)
    plt.imshow(image,cmap=cmap)
    plt.imsave(image_name,image,cmap=cmap)
    #plt.show()


def start_game():
    pyautogui.click(242, 585)
    pyautogui.click(242, 585)


triangle_area = (220,130,1,470)
threshold = 120
fn = lambda x : 255 if x> threshold else 0
resize = 1

def take_screenshot():
    screen = screenshot(triangle_area).convert("L")#.point(fn,mode='1')
    #plot_img(screen,cmap = 'gray')
    array = np.array(screen)
    #print(array.shape, " array shape")
    #print(array)
    text_area = (239,130 + np.where(array >= 80)[0][0] - 6, 240,16)
    screen = screenshot(triangle_area).convert("L")
    array = np.array(screen)
    text_screen = screenshot(text_area).convert("RGB")
    #plot_img(text_screen,cmap='gray')
    return np.array(text_screen)



embedding_dimension = 64
index_filename = 'index'
index = annoy.AnnoyIndex(embedding_dimension,metric = 'angular')
index.load(index_filename)
print('Annoy index is loaded.')
with open(index_filename + '.mapping', 'rb') as handle:
    mapping = pickle.load(handle)
print('Mapping file is loaded.')



model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
print("Loading the TF-Hub model...")
embed_fn = hub.load(model_url)
print("TF-Hub model is loaded.")


def find_similar_items(embedding, num_matches=5):
    '''Finds similar items to a given embedding in the ANN index'''
    ids = index.get_nns_by_vector(
    embedding, num_matches, search_k=-1, include_distances=False)
    items = [mapping[i] for i in ids]
    return items
    
random_projection_matrix = None
if os.path.exists('random_projection_matrix'):
    print("Loading random projection matrix...")
    with open('random_projection_matrix', 'rb') as handle:
        random_projection_matrix = pickle.load(handle)
    print('random projection matrix is loaded.')

def extract_embeddings(query):
    '''Generates the embedding for the query'''
    print(query, ' query')
    query_embedding =  embed_fn([query])[0].numpy()
    print('embed ', query_embedding.shape)

    if random_projection_matrix is not None:
        query_embedding = random_projection_matrix.transform(query_embedding.reshape(1,-1)).reshape(-1,)
        print('projected', query_embedding.shape)
    return query_embedding


pipeline = keras_ocr.pipeline.Pipeline()


def run():
    print()
    print('Begin')
    screenshot = take_screenshot()
    print('take screenshot')
    text,_ = pipeline.recognize([screenshot])[0][0]    
    print(text, ' generated text from ocr')
    query_embedding = extract_embeddings(text)
    print('query_embedding')
    items = find_similar_items(query_embedding, 5)
    print('items')
    if text.startswith(items[0]):
        pyperclip.copy(items[random.randint(1,3)])
    else:
        pyperclip.copy(items[random.randint(0,2)])
    print(items, ' items generated from embeding and ann')
    pyautogui.click(x=100, y=200)
    pyautogui.hotkey('command', 'v')
    pyautogui.press('enter')

if __name__ == "__main__":
    i = 0
    import time
    while i < 20 :
        run()
        time.sleep(3)
        i += 1
        print()
        print()
