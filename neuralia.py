#!/usr/bin/env python3 
# -*- coding: utf-8 -*-"
"""
NeuraLiA 2020 - by psy (epsylon@riseup.net)
"""
import numpy as np

print(75*"=")
print(" _   _            ____       _     ___    _    ")
print("| \ | | ___ _   _|  _ \ __ _| |   |_ _|  / \   ")
print("|  \| |/ _ \ | | | |_) / _` | |    | |  / _ \  ")
print("| |\  |  __/ |_| |  _ < (_| | |___ | | / ___ \ ")
print("|_| \_|\___|\__,_|_| \_\__,_|_____|___/_/   \_| by psy")
print("                                               ")
print(75*"=","\n")
print('"Advanced -SYA (Sigmoid + YFactor_Algorithm)- Neural Network"\n')
print(75*"=")

########################### PoC ###################################
dataset = np.array(([1, 1], [1, 2], [1, 3], [1, 4]), dtype=float) # training set
y_factor = 0.000003141537462295 #  in-time reply Y-factor
y = np.array(([2], [3], [4]), dtype=float) # score results
print("\n + Training Set:\n") # printing output
print("     1 + 1 = 2")
print("     1 + 2 = 3")
print("     1 + 3 = 4")
print("\n + Question:\n")
print("     1 + 4 = ?")
print("\n + Answer (expected):\n")
print("     5\n")
print(75*"=")
########################### PoC ###################################

simulation = input("\nDo you wanna start? (Y/n): ")
if simulation == "n" or simulation == "N":
    import sys
    sys.exit()
dataset = dataset/np.amax(dataset, axis=0)
y = y/100
X = np.split(dataset, [3])[0]
xPredicted = np.split(dataset, [3])[1]

class Neural_Network(object):
  def __init__(self):
    self.inputSize = 2 
    self.hiddenSize = 3
    self.outputSize = 1
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

  def predict(self):
    print("="*75)
    total_neurons = self.inputSize + self.hiddenSize + self.outputSize
    print("-> NEURONS: ["+str(total_neurons)+"] (Input: ["+str(self.inputSize)+"] | Hidden: ["+str(self.hiddenSize)+"] | Output: ["+str(self.outputSize)+"])")
    print("="*75)
    print("\n + Input (scaled): \n\n " + str(xPredicted))
    print("\n + Prediction (scaled): \n\n " + str(self.forward(xPredicted)))
    print("\n + Answer (predicted): \n\n " + str(round(int(self.forward(xPredicted)*100), 2)))
    print("\n"+"-"*50+"\n")

NN = Neural_Network()
t = 0
while True:
  loss = np.mean(np.square(y - NN.forward(X)))
  if loss > y_factor:
      t = t + 1
      print("="*75)
      print("-> ROUNDS (Learning): "+str(t))
      print("="*75)
      print("\n + Input (scaled): \n\n " + str(X).replace("      ",""))
      print("\n + Actual Output: \n\n " + str(y))
      print("\n + Predicted Output: \n\n " + str(NN.forward(X)))
      print("\n + Loss: \n\n [" + str(loss)+"]\n")
      NN.train(X, y)
  else:
      break
NN.predict()
