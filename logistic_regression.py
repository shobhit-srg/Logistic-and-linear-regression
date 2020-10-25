import pandas as pan
import numpy as num
from PIL import Image
import os
import cv2
import sys

train_txt=str(sys.argv[1])
test_txt=str(sys.argv[2])

def unique(list1): 
    x = np.array(list1) 
    return np.unique(x) 
# train_txt=root_path+"sample_train.txt"
# test_txt=root_path+"sample_test.txt"
f = open(train_txt, 'r')
x = f.readlines()
f.close()
f1 = open(test_txt, 'r')
ax = f1.readlines()
f1.close()
for i in range(len(ax)):
	ax[i]=ax[i].rstrip()
label_str=[]
for i in range(len(x)):
	x[i]=x[i].rstrip()
	x[i]=x[i].split(' ')
import numpy as np
for i in range(len(x)):
	label_str.append(x[i][1])
l_str=unique(label_str)
list_label={}
for i in range(len(l_str)):
	list_label[l_str[i]]=i
paths=[]
for i in range(len(x)):
	paths.append(x[i][0])
# print(paths)
# print(label_str)
# print(list_label)
label_int=[]
for i in range(len(label_str)):
  res = list(list_label.keys()).index(label_str[i])
  label_int.append(list(list(list_label.items())[res]))

# print(label_int)
test_l=[]
for i in range(len(label_int)):
  test_l.append(label_int[i][1])

# print(test_l)

  
def rgb2gray(rgb):
    return num.dot(rgb[...,:3], [0.299, 0.587, 0.144])
img_list=[]
img_name=test_l
orig=[]
for i in range(len(paths)):
  x=paths[i]
  img = num.asarray(Image.open(x))
  gray = rgb2gray(img)
  gray=cv2.resize(gray,(50,50))
  # orig.append(gray)
  gray=gray.flatten()
  img_list.append(gray)
img_list=num.array(img_list)
# print(img_list)
# print(num.size(img_list,0))
# print(num.size(img_list,1))
test_img=[]
for i in range(len(ax)):
  x=ax[i]
  img = num.asarray(Image.open(x))
  gray = rgb2gray(img)
  gray=cv2.resize(gray,(50,50))
  # orig.append(gray)
  gray=gray.flatten()
  test_img.append(gray)
test_img=num.array(test_img)

# print(num.size(test_img,0))
# print(num.size(test_img,1))
# print(test_img)


c_m=num.cov(img_list.T)
EigVal,EigVec = num.linalg.eig(c_m)
EigVal=EigVal.real
EigVec=EigVec.real
order = EigVal.argsort()[::-1]
EigVal = EigVal[order]
EigVec = EigVec[:,order]
eig_pairs = [(num.abs(EigVal[i]), EigVec[:,i]) for i in range(len(EigVal))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

mw1=[]
for i in range(300):
  mw1.append(eig_pairs[i][1].reshape(2500,1))
mw=num.hstack(mw1)

Y = img_list.dot(mw)

c_m=num.cov(test_img.T)
EigVal,EigVec = num.linalg.eig(c_m)
EigVal=EigVal.real
EigVec=EigVec.real
order = EigVal.argsort()[::-1]
EigVal = EigVal[order]
EigVec = EigVec[:,order]
eig_pairs = [(num.abs(EigVal[i]), EigVec[:,i]) for i in range(len(EigVal))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

mw1=[]
for i in range(300):
  mw1.append(eig_pairs[i][1].reshape(2500,1))
mw=num.hstack(mw1)
Y1 = test_img.dot(mw)

i123=num.asarray(img_name)
num_train = int(len(img_list))
train_data=Y
test_data=Y1
train_class=i123
# test_class=i123[num_train:]

# print(num.size(train_data,0))
# print(num.size(train_data,1))
# print(num.size(test_data,0))
# print(num.size(test_data,1))
# print(train_class)


key_list = list(list_label.keys()) 
val_list = list(list_label.values()) 

# for i in range(len(train_class)):
# 	print(key_list[val_list.index(train_class[i])]) 
# print(key_list[val_list.index(112)]) 

import numpy as np
epsilon = 1e-5

def cost(theta, X, y):
    m = len(y)
    #print(m)
    h =.5 * (1 + num.tanh(.5 * (X @ theta)))
    epsilon = 1e-5
    cos = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    grad = 1 / m * ((y - h) @ X)
    return cos,grad



def fit(x, y, max_iter=50000, alpha=0.0000001):
    x = num.insert(x, 0, 1, axis=1)
    # print(y)
    thetas = []
    classes = num.unique(y)
    costs = num.zeros(max_iter)
    # print(classes)
    for c in classes:
        binary_y = num.where(y == c, 1, 0)
        theta = num.zeros(x.shape[1])
        for epoch in range(max_iter):
            # print("dsf",epoch)
            costs[epoch], grad = cost(theta, x, binary_y)
            theta += alpha * grad
            
        thetas.append(theta)
    return thetas, classes, costs

thetas, classes, costs = fit(train_data, train_class)

def predict(classes, thetas, x):
    x = np.insert(x, 0, 1, axis=1)
    preds = [np.argmax([.5 * (1 + num.tanh(.5 * (xi @ theta))) for theta in thetas]
    ) for xi in x]
    return preds
    
i133=predict(classes, thetas, test_data);

# print(i133)

for i in range(len(i133)):
	print(key_list[val_list.index(i133[i])]) 
