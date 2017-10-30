#!/bin/python3
import numpy as np
import csv
import itertools

trainfile="train.csv"
testfile="test.csv"
c = []
t = []
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




def derivative(x):
    return (x)*(1.0-(x))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


# https://stackoverflow.com/questions/34968722/softmax-function-python
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     print(e_x.shape)
#     print(e_x.sum(axis=1))
#     exit(0)
#     return e_x / e_x.sum(axis=1)

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def filereader(file,tt):
    with open(file,'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)
        data_list.pop(0)
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                data_list[i][j]=int(data_list[i][j])
    return data_list


def sanitize(data_list):
    outputs=list()
    for i in range(len(data_list)):
        outputs.append((data_list[i][-1]))
        data_list[i].pop()
    return data_list,outputs

def convert_input(x,i):
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

def convert_inputs(input):
    for i in range(len(input)):
        input[i]=convert_input(input[i],i)
    return input

def convert_outputs(output):
    for i in range(len(output)):
        output[i]=fout[output[i]]
    return output

def select_good_dataset(train_data):
    
    cnt = [0,0,0,0,0,0,0,0,0,0]
    cnt_mx = [70000,70000,30000,30000,30000,30000,30000,30000,30000,30000]
    new_dataset = []
    
    for i in range(len(train_data)):
        for class_no in range(10):
            if train_data[i][-1] == class_no:
                if cnt[class_no] <= cnt_mx[class_no]:
                    cnt[class_no] = cnt[class_no] + 1
                    new_dataset.append(train_data[i])

    print(len(new_dataset))

    return new_dataset







# def pre_process(train_data):
#     temp_train = []
#     for i in range(len(train_data)):
#         if train_data[i][-1] > 1:
#             c = train_data[i][-1]

#             temp = []
#             j = 0
#             for k in range(5):
#                 temp.append( [ train_data[i][j] , train_data[i][j+1] ] )
#                 j = j + 2

#             p = list(itertools.permutations(temp))
#             # p = [ ([s1,c1] , [s2,c2] , [s3,c3] , [s4,c4] , [s5,c5]) ,  ([s1,c1] , [s2,c2] , [s3,c3] , [s4,c4] , [s5,c5]) ]
#             temp_train = []
#             for k in range(len(p)):
#                 for m in range(len(p[k])):
#                     temp_train.append(p[k][m][0])
#                     temp_train.append(p[k][m][1])

#                 temp.append(c)
#                 del train_data[i]
#                 train_data.append(temp)
           







def convert_back(output):

    for i in range(len(output)):
        
        k = output[i].argmax()
        
        # print(output[i][k])
        # print(output[i])
        # exit(0)

        output[i] = [k]

        # for j in range(10):

        #     if j == k:
        #         output[i][j] = 1
        #     else:
        #         output[i][j] = 0

        #     # if(output[i][j]>.5):
        #     #     output[i][j]=1
        #     # else:
        #     #     output[i][j]=0
    
        # for j in range(10):
        #     if (output[i]==np.array(fout[j])).all():
        #         output[i]=[j]
        #         break

    return output

# def remove_some_rows(x):
    
#     cnt = 0
    
#     for i in range(len(x)):
#         if (x[i][-1] == 0) or (x[i][-1] == 1):
#             cnt = cnt + 1
#     return (cnt,len(x))


train_data = filereader(trainfile,False)

train_data = select_good_dataset(train_data)


train_data = np.array(train_data)

np.random.shuffle(train_data)


# data augmentation
# train_data = pre_process(train_data) 

train_data = train_data.tolist()

# cnt,n = remove_some_rows(train_data)

# print(cnt[0])
# print(cnt[1])

X,y = sanitize(train_data)

X=convert_inputs(X)
y=convert_outputs(y)

# X = X[0:20000]
# y = y[0:20000]


for i in range(len(X)):
    X[i]=np.array(X[i])
    y[i]=np.array(y[i])

X=np.array(X)
y=np.array(y)





test_input = filereader(testfile,True)

test_input = convert_inputs(test_input)

for i in range(len(test_input)):
    test_input[i]=np.array(test_input[i])
test_input = np.array(test_input)

dim1 = len(X[0])

dim2 = 80
dim3 = 100
dim4 = 100
dim5 = 60
dim6 = 10

np.random.seed(1)

W1 = 2*np.random.random((dim1,dim2))-1
W2 = 2*np.random.random((dim2,dim3))-1
W3 = 2*np.random.random((dim3,dim4))-1
W4 = 2*np.random.random((dim4,dim5))-1
W5 = 2*np.random.random((dim5,dim6))-1


b1 = np.array([np.random.random(dim2)])

b2 = np.array([np.random.random(dim3)])

b3 = np.array([np.random.random(dim4)])

b4 = np.array([np.random.random(dim5)])

b5 = np.array([np.random.random(dim6)])

eta = 0.0001

for j in range(150):

    # print(j)

    for k in range(len(X)):
        
        a0 = np.array([X[k]])

        a1 = sigmoid(np.dot(a0,W1) + b1)
        #a1 = sigmoid(np.dot(a0,W1))

        a2 = sigmoid(np.dot(a1,W2) + b2)
        #a2 = sigmoid(np.dot(a1,W2))
        
        a3 = sigmoid(np.dot(a2,W3) + b3)
        #a3 = softmax(np.dot(a2,W3))
        
        a4 = sigmoid(np.dot(a3,W4) + b4)
        #a4 = sigmoid(np.dot(a3,W4))
        
        a5 = softmax(np.dot(a4,W5) + b5)
        # exit(0)

        
        #del1 = -np.array([y[k]]) * (1 - a3)
        del1 = np.array([y[k]]) - a5
        #del1 = ( np.array([y[k]]) - a3 ) * derivative(a3)
        del2 =  del1.dot(W5.T) * derivative(a4)  
        del3 =  del2.dot(W4.T) * derivative(a3)  
        del4 =  del3.dot(W3.T) * derivative(a2)
        del5 =  del4.dot(W2.T) * derivative(a1) 


        W5 += eta*a4.T.dot(del1)
        b5 += eta*del1

        W4 += eta*a3.T.dot(del2)
        b4 += eta*del2

        W3 += eta*a2.T.dot(del3)
        b3 += eta*del3
        
        W2 += eta*a1.T.dot(del4)
        b2 += eta*del4
        
        W1 += eta*a0.T.dot(del5)
        b1 += eta*del5


a0 = test_input

# print(a0.shape)
# exit(0)

a1 = sigmoid(np.dot(a0,W1) + b1)
a2 = sigmoid(np.dot(a1,W2) + b2)
a3 = sigmoid(np.dot(a2,W3) + b3)
a4 = sigmoid(np.dot(a3,W4) + b4)
a5 = softmax(np.dot(a4,W5) + b5)

# a1 = sigmoid(np.dot(a0,W1))
# a2 = sigmoid(np.dot(a1,W2))
# a3 = softmax(np.dot(a2,W3))
# #a4 = sigmoid(np.dot(a3,W4))

print ("id,predicted_class")
# a4 = convert_back(a4)
a5 = convert_back(a5)
for x in range(len(a5)):
    print (",".join([str(int(x)),str(int(a5[x][0]))]))








# ========================================
# ========================================







# X = X[0:20000]
# y = y[0:20000]


# test_input = filereader(testfile,True)

# test_input = convert_inputs(test_input)

# for i in range(len(test_input)):
#     test_input[i]=np.array(test_input[i])
# test_input = np.array(test_input)

# dim1 = len(X[0])

# dim2 = 23
# dim3 = 20
# dim4 = 25
# dim5 = 10

# np.random.seed(1)

# W1 = 2*np.random.random((dim1,dim2))-1
# W2 = 2*np.random.random((dim2,dim3))-1
# W3 = 2*np.random.random((dim3,dim4))-1
# W4 = 2*np.random.random((dim4,dim5))-1


# b1 = np.array([np.random.random(dim2)])

# b2 = np.array([np.random.random(dim3)])

# b3 = np.array([np.random.random(dim4)])

# b4 = np.array([np.random.random(dim5)])

# eta=0.001

# for j in range(20):

#     # print(j)

#     for k in range(len(X)):
        
#         a0 = np.array([X[k]])

#         a1 = sigmoid(np.dot(a0,W1) + b1)
#         #a1 = sigmoid(np.dot(a0,W1))

#         a2 = sigmoid(np.dot(a1,W2) + b2)
#         #a2 = sigmoid(np.dot(a1,W2))
        

#         a3 = sigmoid(np.dot(a2,W3) + b3)
#         #a3 = sigmoid(np.dot(a2,W3))
        
#         a4 = sigmoid(np.dot(a3,W4) + b4)
#         #a4 = sigmoid(np.dot(a3,W4))
        
        


#         # exit(0)

        
#         del1 = -np.array([y[k]]) * (1 - a4)
#         del2 =  del1.dot(W4.T) * derivative(a3)  
#         del3 =  del2.dot(W3.T) * derivative(a2)  
#         del4 =  del3.dot(W2.T) * derivative(a1)  



#         W4 -= eta*a3.T.dot(del1)
#         b4 -= eta*del1

#         W3 -= eta*a2.T.dot(del2)
#         b3 -= eta*del2
        
#         W2 -= eta*a1.T.dot(del3)
#         b2 -= eta*del3
        
#         W1 -= eta*a0.T.dot(del4)
#         b1 -= eta*del4


# a0 = test_input

# # print(a0.shape)
# # exit(0)

# a1 = sigmoid(np.dot(a0,W1) + b1)
# a2 = sigmoid(np.dot(a1,W2) + b2)
# a3 = sigmoid(np.dot(a2,W3) + b3)
# a4 = sigmoid(np.dot(a3,W4) + b4)

# # a1 = sigmoid(np.dot(a0,W1))
# # a2 = sigmoid(np.dot(a1,W2))
# # a3 = sigmoid(np.dot(a2,W3))
# # a4 = sigmoid(np.dot(a3,W4))

# print ("id,predicted_class")
# a4 = convert_back(a4)
# for x in range(len(a4)):
#     print (",".join([str(int(x)),str(int(a4[x][0]))]))