import numpy as np
import csv
import pandas as pd
#files for input and output
trainfile="train.csv"
testfile="test.csv"
learning_rate=0.1

global total_layers
global opLayerNo
global dimLi, layerLi, weightLi, layerErLi, layerDeLi, softmaxOp
global X,y
    

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

#########################
##  PREPROCESS  DATA   ##
#########################
#Embedding

#separates input and output
def sanitize(data_list):
    outputs=list()
    for i in range(len(data_list)):
        outputs.append((data_list[i][-1]))
        data_list[i].pop()
    return data_list,outputs


def preprocess_data():
    global X,y, test_input
    train_data=filereader(trainfile)
    X,y=sanitize(train_data)

    X=embed_input(X)
    y=embed_output(y)
    for i in range(len(X)):
        X[i]=np.array(X[i])
        y[i]=np.array(y[i])
    X=np.array(X)
    y=np.array(y)

    test_input=filereader(testfile)

    test_input = embed_input(test_input)
    for i in range(len(test_input)):
        test_input[i]=np.array(test_input[i])

    test_input=np.array(test_input)

#reads the file and returns the data in raw form
def filereader(file):
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


def embed_input(ip):
    for i in range(len(ip)):
            ip[i]=convert_input(ip[i])
    return ip

def embed_output(output):
    for i in range(len(output)):
        output[i]=fout[output[i]]
    return output

def convert_back(output):
    final_op = np.zeros(len(output))
    for i in range(len(output)):
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


preprocess_data()
#------------------------------------------------------------------preprocessing/embedding ends here


#######################
##   ACTIVATION      ##
#######################

#Softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#Function differentiation
def sigmoid_prime(x):
    return (x)*(1.0-(x))

#Function Evaluation
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def forward_prop(k):

    ##########################
    ## FORWARD  PROPAGATION ##
    ##########################


    global total_layers
    global opLayerNo
    global dimLi, layerLi, weightLi, layerErLi, layerDeLi, softmaxOp, softmaxLayer
    global X
    layerLi[0] = np.array([X[k]])
    for x in range(1,total_layers):
        layerLi[x] = sigmoid(np.dot(layerLi[x-1],weightLi[x-1]))

    layerLi[softmaxLayer] = softmax(layerLi[opLayerNo])
    



def cost_comp(k):

    ##########################
    ## cost computation     ##
    ##########################


    global total_layers
    global opLayerNo
    global dimLi, layerLi, weightLi, layerErLi, layerDeLi, softmaxOp, softmaxLayer
    
    return layerLi[softmaxLayer]-np.array([y[k]])


def back_prop(k):

    ##########################
    ## backward propagation ##
    ##########################

    global total_layers
    global opLayerNo
    global dimLi, layerLi, weightLi, layerErLi, layerDeLi, softmaxOp, softmaxLayer
    layerErLi = [0]*total_layers
    layerDeLi = [0]*total_layers   
    
    layerErLi[opLayerNo] = cost_comp(k)
    layerDeLi[opLayerNo] = layerErLi[opLayerNo]*sigmoid_prime(layerLi[opLayerNo])
    
    for x in range(opLayerNo-1, 0, -1):
        layerErLi[x] = layerDeLi[x+1].dot(weightLi[x].T)
        layerDeLi[x] = layerErLi[x]*sigmoid_prime(layerLi[x])

def param_update():

    #########################
    ## PARAMETER UPDATION  ##
    #########################

    global total_layers
    global opLayerNo
    global dimLi, layerLi, weightLi, layerErLi, layerDeLi,softmaxOp, softmaxLayer

    for x in range(opLayerNo-1,-1,-1):
        weightLi[x] += -learning_rate*layerLi[x].T.dot(layerDeLi[x+1])

def model(hidden_layers_no):

    #########################
    ##        MODEL        ##
    #########################

    global total_layers
    global opLayerNo
    global dimLi, layerLi, weightLi, layerErLi, layerDeLi, softmaxOp, softmaxLayer
    global X,y, test_input

    total_layers = hidden_layers_no+2
    ipLayerNo = 0
    opLayerNo = hidden_layers_no+1
    softmaxLayer = opLayerNo+1
    dimLi = []
    layerLi = [0]*(total_layers+1)


    dimLi.append(len(X[0]))
    dimLi.append(18)
    for x in range(hidden_layers_no-1):
        dimLi.append(10)
        
    dimLi.append(10)

    np.random.seed(1)
    weightLi = []
    for x in range(hidden_layers_no+1):
        weightLi.append(2*np.random.random((dimLi[x],dimLi[x+1]))-1)

    #train the network
    #Stochastic Gradient Descent
    for j in range(5):
        print(j)
        for k in range(len(X)):
            forward_prop(k);
            back_prop(k);
            param_update()

    layerLi[0] = test_input 
    for x in range(1,total_layers):
        layerLi[x] = sigmoid(np.dot(layerLi[x-1],weightLi[x-1]))

    layerLi[softmaxLayer] = softmax(layerLi[opLayerNo])


    layer_fin=convert_back(layerLi[softmaxLayer])
    arr = layer_fin.astype(int)
    id_ar = np.arange(1,arr.size+1)
    np.savetxt("output.csv", np.dstack((np.arange(0, arr.size),arr))[0],"%d,%d",header="id,predicted_class",comments='')

model(2)
