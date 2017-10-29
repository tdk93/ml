import numpy as np
import csv
import pandas as pd
#files for input and output
trainfile="train.csv"
testfile="test.csv"
learning_rate=0.1

c=[]
t=[]
fout=[]
l=[1,0,0,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,1,0,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,1,0,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,1,0,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,1,0,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,1,0,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,1,0,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,1,0,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,1,0,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,1,0,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,1,0,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,0,1,0]
c.append(l)
l=[0,0,0,0,0,0,0,0,0,0,0,0,1]
c.append(l)
l=[1,0,0,0]
t.append(l)
l=[0,1,0,0]
t.append(l)
l=[0,0,1,0]
t.append(l)
l=[0,0,0,1]
t.append(l)
l=[1,0,0,0,0,0,0,0,0,0]
fout.append(l)
l=[0,1,0,0,0,0,0,0,0,0]
fout.append(l)
l=[0,0,1,0,0,0,0,0,0,0]
fout.append(l)
l=[0,0,0,1,0,0,0,0,0,0]
fout.append(l)
l=[0,0,0,0,1,0,0,0,0,0]
fout.append(l)
l=[0,0,0,0,0,1,0,0,0,0]
fout.append(l)
l=[0,0,0,0,0,0,1,0,0,0]
fout.append(l)
l=[0,0,0,0,0,0,0,1,0,0]
fout.append(l)
l=[0,0,0,0,0,0,0,0,1,0]
fout.append(l)
l=[0,0,0,0,0,0,0,0,0,1]
fout.append(l)

X = 0
y = 0
test_input = 0
# PREPROCESS  DATA
##
#Embedding
def preprocess_data():
    global X,y, test_input
    train_data=filereader(trainfile,False)
    X,y=sanitize(train_data)

    X=embed_input(X)
    y=embed_output(y)
    for i in range(len(X)):
        X[i]=np.array(X[i])
        y[i]=np.array(y[i])
    X=np.array(X)
    y=np.array(y)

    test_input=filereader(testfile,True)

    test_input = embed_input(test_input)
    for i in range(len(test_input)):
        test_input[i]=np.array(test_input[i])

    test_input=np.array(test_input)
    #test_input=test_input[0:10]

#reads the file and returns the data in raw form
def filereader(file,tt):
    with open(file,'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        #might need to pop the header
        data_list.pop(0)
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                data_list[i][j]=int(data_list[i][j])
    return data_list


#converts inputs and outputs
def convert_input(x):
    ans=[]
    ans.extend(t[x[0]-1])
    ans.extend(c[x[1]-1])
    ans.extend(t[x[2]-1])
    ans.extend(c[x[3]-1])
    ans.extend(t[x[4]-1])
    ans.extend(c[x[5]-1])
    ans.extend(t[x[6]-1])
    ans.extend(c[x[7]-1])
    ans.extend(t[x[8]-1])
    ans.extend(c[x[9]-1])
    return ans


def embed_input(input):
    for i in range(len(input)):
            input[i]=convert_input(input[i])
    return input

def embed_output(output):
    for i in range(len(output)):
        output[i]=fout[output[i]]
    return output

def convert_back(output):
    final_op = np.zeros(len(output))
    for i in range(len(output)):
        #print("i = ", i)
        maxv = 0
        maxi = 0
        for j in range(10):
            if(output[i][j] > maxv):
                maxv = output[i][j]
                maxi = j


        for m in range(10):
            output[i][m] = int(0)
        output[i][maxi] = int(1)
        for j in range(10):
            if (output[i]==np.array(fout[j])).all():
                final_op[i] = int(j)
                output[i]=[j]
                break
    return final_op 

#separates input and output
def sanitize(data_list):
    outputs=list()
    for i in range(len(data_list)):
        outputs.append((data_list[i][-1]))
        data_list[i].pop()
    return data_list,outputs


preprocess_data()
#------------------------------------------------------------------preprocessing/embedding ends here


#######################
##   ACTIVATION      ##
#######################


#Function differentiation
def sigmoid_prime(x):
    return (x)*(1.0-(x))

#Function Evaluation
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def forward_prop():
    global layer_0, layer_1, layer_2, layer_3, layer_1_error, layer_1_delta, layer_2_error, layer_2_delta, layer_3_error, layer_3_delta, layer_3_fin
    global weight0, weight1, weight2
 
    layer_0 = np.array([X[k]])
    layer_1 = sigmoid(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1,weight1))
    layer_3 = sigmoid(np.dot(layer_2,weight2))


def cost_comp():
    global layer_0, layer_1, layer_2, layer_3, layer_1_error, layer_1_delta, layer_2_error, layer_2_delta, layer_3_error, layer_3_delta, layer_3_fin
    global weight0, weight1, weight2
    #BACKWARD PROPAGATION
    return np.array([y[k]]) - layer_3


def back_prop():
    global layer_0, layer_1, layer_2, layer_3, layer_1_error, layer_1_delta, layer_2_error, layer_2_delta, layer_3_error, layer_3_delta, layer_3_fin
    global weight0, weight1, weight2
    #BACKWARD PROPAGATION
    layer_3_error = cost_comp() 
    layer_3_delta = layer_3_error * sigmoid_prime(layer_3)

    layer_2_error = layer_3_delta.dot(weight2.T)
    layer_2_delta = layer_2_error * sigmoid_prime(layer_2)

    layer_1_error = layer_2_delta.dot(weight1.T)
    layer_1_delta = layer_1_error * sigmoid_prime(layer_1)

def param_update():
    #PARAMETER UPDATION
    global layer_0, layer_1, layer_2, layer_3, layer_1_error, layer_1_delta, layer_2_error, layer_2_delta, layer_3_error, layer_3_delta, layer_3_fin
    global weight0, weight1, weight2
    weight2 += learning_rate*layer_2.T.dot(layer_3_delta)
    weight1 += learning_rate*layer_1.T.dot(layer_2_delta)
    weight0 += learning_rate*layer_0.T.dot(layer_1_delta)



def train_nw():
    global X,y, test_input
    dim1 = len(X[0])
    dim2 = 18
    dim3 = 10
    dim4 = 10
    np.random.seed(1)
    #weight vectors
    weight0 = 2*np.random.random((dim1,dim2))-1
    weight1 = 2*np.random.random((dim2,dim3))-1
    weight2 = 2*np.random.random((dim3,dim4))-1

    #train the network
    #Stochastic Gradient Descent
    for j in range(5):
        print(j)
        for k in range(len(X)):
            forward_prop();
            back_prop()

    layer_0 = test_input
    layer_1 = sigmoid(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1,weight1))
    layer_3 = sigmoid(np.dot(layer_2,weight2))
    layer_3_fin=convert_back(layer_3)
    arr = layer_3_fin.astype(int)
    id_ar = np.arange(1,arr.size+1)
    np.savetxt("output.csv", np.dstack((np.arange(0, arr.size),arr))[0],"%d,%d",header="id,predicted_class",comments='')

train_nw()

