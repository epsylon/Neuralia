#!/usr/bin/env python3 
# -*- coding: utf-8 -*-"
"""
NeuraLiA 2020 - by psy (epsylon@riseup.net)
"""
import time
import glob
import os.path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array
from scipy.ndimage.interpolation import zoom
from sklearn.cluster import KMeans
from skimage import measure

VERSION = "v:0.2beta"
RELEASE = "21052020"
SOURCE1 = "https://code.03c8.net/epsylon/neuralia"
SOURCE2 = "https://github.com/epsylon/neuralia"
CONTACT = "epsylon@riseup.net - (https://03c8.net)"

print(75*"=")
print(" _   _            ____       _     ___    _    ")
print("| \ | | ___ _   _|  _ \ __ _| |   |_ _|  / \   ")
print("|  \| |/ _ \ | | | |_) / _` | |    | |  / _ \  ")
print("| |\  |  __/ |_| |  _ < (_| | |___ | | / ___ \ ")
print("|_| \_|\___|\__,_|_| \_\__,_|_____|___/_/   \_| by psy")
print("                                               ")
print(75*"=","\n")
print('"Advanced Recognition {S.Y.A.} Neural Network"\n')
print("\n"+"-"*15+"\n")
print(" * VERSION: ")
print("   + "+VERSION+" - (rev:"+RELEASE+")")
print("\n * SOURCES:")
print("   + "+SOURCE1)
print("   + "+SOURCE2)
print("\n * CONTACT: ")
print("   + "+CONTACT+"\n")
print("-"*15+"\n")
print("="*50)

simulation = input("\nDo you wanna start? (Y/n): ")
if simulation == "n" or simulation == "N":
    import sys
    sys.exit()

memo_path = "stored/memo.dat"
input_neurons = 3
hidden_neurons = 62
output_neurons = 1

class Questioning_Network(object):
    def make_question(self, images_dataset):
        concepts_dataset = {}
        i = 0
        for k, v in images_dataset.items():
            i = i + 1
            img = Image.fromarray(v, 'RGB')
            img.show()
            answer = input("\n[AI] -> Asking: 'What do you think is this? (ex: building)' -> inputs/"+str(k)+"\n")
            print("\n[You]-> Replied: "+str(answer))
            concepts_dataset[i] = str(answer)
            with open(memo_path, 'w') as data:
                data.write(str(concepts_dataset))
            print("[AI] -> Memorizing ...")
            time.sleep(1)
        print("")

class Conceptual_Network(object):
    def extract_concepts(self):
        concepts_dataset = {}
        f = open(memo_path, "r")
        concepts_dataset = f.read()
        f.close()
        return concepts_dataset

class Visual_Network(object):
    def __init__(self):
        print("\n"+"="*75+"\n")
        images = glob.glob("inputs/*")
        for image in images:
            image = image.replace("inputs/","")
            if not os.path.isfile('outputs/'+image):
                im = plt.imread("inputs/"+image)
                print("[AI] -> Visualizing: "+str(image))
                im_small = zoom(im, (1,0,1))
                h,w = im.shape[:2]
                im_small_long = im.reshape((h * w, 3))
                im_small_wide = im.reshape((h,w,3))
                km = KMeans(n_clusters=3)
                km.fit(im_small_long)
                cc = km.cluster_centers_.astype(np.uint8)
                out = np.asarray([cc[i] for i in km.labels_]).reshape((h,w,3))
                seg = np.asarray([(1 if i == 1 else 0)
                for i in km.labels_]).reshape((h,w))
                contours = measure.find_contours(seg, 0.4, fully_connected="high")
                simplified_contours = [measure.approximate_polygon(c, tolerance=4) for c in contours]
                plt.figure(figsize=(5,10))
                for n, contour in enumerate(simplified_contours):
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)       
                plt.ylim(h,0)
                print("[AI] -> Analyzing: "+str(image))
                plt.axis('off')
                plt.savefig('outputs/'+image, bbox_inches='tight', transparent=True, pad_inches=0)
            else:
                print("[AI] -> Recognizing: "+str(image))

    def extract_dataset(self):
        images_dataset = {}
        images = glob.glob("inputs/*")
        for image in images:
            image = image.replace("inputs/","")
            print("[AI] -> Remembering: "+str(image))
            img = Image.open('outputs/'+image)
            dataset = np.array(img)
            images_dataset[image] = dataset
        return images_dataset

class Neural_Network(object):
    def __init__(self):
        self.inputSize = input_neurons
        self.hiddenSize = hidden_neurons
        self.outputSize = output_neurons
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1) 
        self.z2 = self.sigmoid(self.z) 
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        return s*(1-s)

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self, dataset, results):
        score = self.forward(xPredicted)
        self.final_reply(score, dataset, results)

    def final_reply(self, score, image, results):
        concepts_dataset = { line.split()[0] : line.split()[1] for line in open(memo_path) }
        for k, v in concepts_dataset.items():
            if "{" in k:
                k = k.replace("{", "")
            if ":" in k:
                k = k.replace(":", "")
            if int(k) == int(score[0]):
                if "," in v:
                    v = v.replace(",", "")
                results[str(image)] = str(v)
            else:
                results[str(image)] = "UNKNOWN ..."

# procedural (high) sub-class
NN = Neural_Network()
VN = Visual_Network()
CN = Conceptual_Network()
QN = Questioning_Network()

# procedural (low) sub-class
images_dataset = VN.extract_dataset()
concepts_dataset = CN.extract_concepts()

if not concepts_dataset:
    QN.make_question(images_dataset)
else:
    t = 0
    prev_loss = None
    results = {}
    for k, v in images_dataset.items():
        dataset = v/np.amax(v, axis=0)
        for data in dataset: 
            X = data
            xPredicted = X
            y_factor = 0.0001 # reply condition factor
            i = 0
            for concept in concepts_dataset:
                i = i + 1
                y = np.array([i], dtype=int)
                while True:
                    loss = np.mean(np.square(y - NN.forward(X)))
                    if prev_loss == loss:
                        break
                    else:
                        if loss > y_factor:
                            t = t + 1
                            print("\n"+"="*75)
                            print("-> ROUNDS (Learning): "+str(t))
                            print("="*75)
                            print("\n + Question (image): inputs/" + str(k))
                            print(" + Current Answer: " + str(y))
                            print(" + Loss: [" + str(loss)+"]")
                            print(" + YFactor: ["+ str(y_factor)+"]")
                            NN.train(X, y)
                            prev_loss = loss
                            y_factor = y_factor + 0.0010
                        else:
                            break
        NN.predict(k, results)
    print("\n"+"="*75)
    total_neurons = input_neurons + hidden_neurons + output_neurons
    print("-> NEURONS: ["+str(total_neurons)+"] (Input: ["+str(input_neurons)+"] | Hidden: ["+str(hidden_neurons)+"] | Output: ["+str(output_neurons)+"])")
    print("="*75)
    print("\n"+"[AI] -> Replying ...\n")
    for k, v in results.items():
        print("   + Image: inputs/"+str(k)+" -> "+str(v))
    print("\n"+"="*75)

