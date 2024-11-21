import sys
import os
import matplotlib.pyplot as plt
import numpy as np
 
 
class nn_linear_layer:
    
    # linear layer.
    # randomly initialized by creating matrix W and bias b
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0,std,(output_size,input_size))
        self.b = np.random.normal(0,std,(output_size,1))

    ######
    ## Q1
    def forward(self,x):
        res = np.matmul(self.W,x.T) + self.b
        res = res.T
        # res is 20,4
        return res
    
   
    ######
    ## Q2
    ## returns three parameters
    def backprop(self,x,dLdy):
        dLdx = np.matmul(dLdy,self.W)
        dLdW = np.matmul(dLdy.T,x) # it becomes 2,2!
        dLdb = np.reshape(np.mean(dLdy, axis = 0),(1,-1))
        return dLdW,dLdb,dLdx

    def update_weights(self,dLdW,dLdb):
        # parameter update
        self.W=self.W+dLdW  
        self.b=self.b+dLdb

class nn_activation_layer:
    
    def __init__(self):
        pass
    
    ######
    ## Q3
    def forward(self,x):
        # overflow 방지 위해 x가 양수일 때는 일반적인 sigmoid, 아닌 경우 분자 분모에 모두 exp(x)곱하기.
        #res = np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
        res = 1 / (1 + np.exp(-x))
        #breakpoint()
        return res
    
    ######
    ## Q4
    def backprop(self,x,dLdy):
        # jacobian of sigmoid function is diagonal. not using matmul -> just elment-wisely multiplicate!
        # x is (20,4)
        # dLdy is (1,4) jacobian
        sigmoid = self.forward(x)
        derivative = (1 - sigmoid) * sigmoid
        dLds1 = dLdy * derivative
        return dLds1


class nn_softmax_layer:
    def __init__(self):
        pass
    ######
    ## Q5
    def forward(self,x):
        # x should be (20,2)
        res = []
        for i in range(0,len(x)):
            denominator = 0
            row = []
            x_max = np.max(x[i])
            # x[i] is (2,)
            # prepare for potential overflow
            for j in range(0,len(x[i])):
                denominator += np.exp(x[i][j]-x_max)
            for j in range(0,len(x[i])):
                row.append(np.exp(x[i][j]-x_max) / denominator)
            res.append(row)
        res = np.array(res)
        return res
    
    ######
    ## Q6
    def backprop(self,x,dLdp):
        # dLdp -> 2차원 row vector   (1,2)
        dLds2 = []
        for i in range(0,len(x)):
            s1 = x[i][0]
            s2 = x[i][1]
            p1 = np.exp(s1) / (np.exp(s1)+np.exp(s2))
            p2 = 1 - p1 # because we have only two p_i
            dpds2 = np.array( [[ p1*(1-p1), -p1*p2 ],\
                               [-p1*p2 , p2*(1-p2)]])
            #for j in range(0,len(x[i])):
            dLds2.append(np.matmul(np.reshape(dLdp[i],(1,-1)),dpds2))
            #dLds2.append(row)
        dLds2 = np.array(dLds2)
        dLds2 = np.reshape(dLds2,(len(x),len(x[i])))
        return dLds2

class nn_cross_entropy_layer:
    def __init__(self):
        pass
        
    ######
    ## Q7
    def forward(self,x,y):
        cross_entropy = []
        for i in range(0,len(x)): # loop : batch size
            b_i = y[i][0]
            p_i = x[i][0]
            c_e = (1- b_i) * np.log(p_i) + (b_i) * np.log(1 - p_i)
            cross_entropy.append([-c_e])
        cross_entropy = np.array(cross_entropy).mean(axis = 0)
        return cross_entropy[0]
        
    ######
    ## Q8
    def backprop(self,x,y):
        dLdp = []
        for i in range(0,len(x)):
            b_i = y[i][0]
            p_i = x[i][0] 
            jacobian_L = [ -( (1 - b_i) / p_i), -(b_i / (1-p_i)) ]
            dLdp.append(jacobian_L)
        dLdp = np.array(dLdp)
        dLdp = np.reshape(dLdp,(len(x),len(x[0])))
        return dLdp

# number of data points for each of (0,0), (0,1), (1,0) and (1,1)
num_d=5

# number of test runs
num_test=40

## Q9. Hyperparameter setting
## learning rate (lr)and number of gradient descent steps (num_gd_step)
## This part is not graded (there is no definitive answer).
## You can set this hyperparameters through experiments.
lr= 0.01
num_gd_step= 100000

# dataset size
batch_size=4*num_d

# number of classes is 2
num_class=2

# variable to measure accuracy
accuracy=0

# set this True if want to plot training data
show_train_data=False

# set this True if want to plot loss over gradient descent iteration
show_loss=True

################
# create training data
################

m_d1 = (0, 0)
m_d2 = (1, 1)
m_d3 = (0, 1)
m_d4 = (1, 0)

sig = 0.05
s_d1 = sig ** 2 * np.eye(2)

d1 = np.random.multivariate_normal(m_d1, s_d1, num_d)
d2 = np.random.multivariate_normal(m_d2, s_d1, num_d)
d3 = np.random.multivariate_normal(m_d3, s_d1, num_d)
d4 = np.random.multivariate_normal(m_d4, s_d1, num_d)

# training data, and has shape (4*num_d,2)
x_train_d = np.vstack((d1, d2, d3, d4))
# training data lables, and has shape (4*num_d,1)
y_train_d = np.vstack((np.zeros((2 * num_d, 1), dtype='uint8'), np.ones((2 * num_d, 1), dtype='uint8')))

if (show_train_data):
    plt.grid()
    plt.scatter(x_train_d[range(2 * num_d), 0], x_train_d[range(2 * num_d), 1], color='b', marker='o')
    plt.scatter(x_train_d[range(2 * num_d, 4 * num_d), 0], x_train_d[range(2 * num_d, 4 * num_d), 1], color='r',
                marker='x')
    plt.show()

################
# create layers
################

# hidden layer
# linear layer
layer1 = nn_linear_layer(input_size=2, output_size=4, )
# activation layer
act = nn_activation_layer()

# output layer
# linear
layer2 = nn_linear_layer(input_size=4, output_size=2, )
# softmax
smax = nn_softmax_layer()
# cross entropy
cent = nn_cross_entropy_layer()

# variable for plotting loss
loss_out = np.zeros((num_gd_step))

################
# do training
################

for i in range(num_gd_step):
    
    # fetch data
    x_train = x_train_d
    y_train = y_train_d
    ################
    # forward pass
    
    # hidden layer
    # linear
    l1_out = layer1.forward(x_train)
    # activation
    a1_out = act.forward(l1_out)
    
    # output layer
    # linear
    l2_out = layer2.forward(a1_out)
    # softmax
    smax_out = smax.forward(l2_out)
    # cross entropy loss
    loss_out[i] = cent.forward(smax_out, y_train)
    ################
    # perform backprop
    # output layer
    # cross entropy
    b_cent_out = cent.backprop(smax_out, y_train)
    # softmax
    b_nce_smax_out = smax.backprop(l2_out, b_cent_out)
    # linear
    b_dLdW_2, b_dLdb_2, b_dLdx_2 = layer2.backprop(x=a1_out, dLdy=b_nce_smax_out)
    
    # backprop, hidden layer
    # activation
    b_act_out = act.backprop(x=l1_out, dLdy=b_dLdx_2)
    #breakpoint()
    # linear
    b_dLdW_1, b_dLdb_1, b_dLdx_1 = layer1.backprop(x=x_train, dLdy=b_act_out)
    #breakpoint()
    
    ################
    # update weights: perform gradient descent
    layer2.update_weights(dLdW=-b_dLdW_2 * lr, dLdb=-b_dLdb_2.T * lr)
    layer1.update_weights(dLdW=-b_dLdW_1 * lr, dLdb=-b_dLdb_1.T * lr)

    """if i == num_gd_step // 2 :
        breakpoint()
"""
    if (i + 1) % 2000 == 0:
        print('gradient descent iteration:', i + 1)

# set show_loss to True to plot the loss over gradient descent iterations
if (show_loss):
    plt.figure(1)
    plt.grid()
    plt.plot(range(num_gd_step), loss_out)
    plt.xlabel('number of gradient descent steps')
    plt.ylabel('cross entropy loss')
    plt.show()

################
# training done
# now testing

num_test = 100

for j in range(num_test):
    
    predicted = np.ones((4,))
    
    # dispersion of test data
    sig_t = 1e-2
    
    # generate test data
    # generate 4 samples, each sample nearby (1,1), (0,0), (1,0), (0,1) respectively
    t11 = np.random.multivariate_normal((1,1), sig_t**2*np.eye(2), 1)
    t00 = np.random.multivariate_normal((0,0), sig_t**2*np.eye(2), 1)
    t10 = np.random.multivariate_normal((1,0), sig_t**2*np.eye(2), 1)
    t01 = np.random.multivariate_normal((0,1), sig_t**2*np.eye(2), 1)
    
    # predicting label for test sample nearby (1,1)
    l1_out = layer1.forward(t11)
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[0] = np.argmax(smax_out)
    print('softmax out for (1,1)', smax_out, 'predicted label:', int(predicted[0]))
    
    # predicting label for test sample nearby (0,0)
    l1_out = layer1.forward(t00)
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[1] = np.argmax(smax_out)
    print('softmax out for (0,0)', smax_out, 'predicted label:', int(predicted[1]))
    
    # predicting label for test sample nearby (1,0)
    l1_out = layer1.forward(t10)
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[2] = np.argmax(smax_out)
    print('softmax out for (1,0)', smax_out, 'predicted label:', int(predicted[2]))
    
    # predicting label for test sample nearby (0,1)
    l1_out = layer1.forward(t01)
    a1_out = act.forward(l1_out)
    l2_out = layer2.forward(a1_out)
    smax_out = smax.forward(l2_out)
    predicted[3] = np.argmax(smax_out)
    print('softmax out for (0,1)', smax_out, 'predicted label:', int(predicted[3]))
    
    print('total predicted labels:', predicted.astype('uint8'))
    
    accuracy += (predicted[0] == 0) & (predicted[1] == 0) & (predicted[2] == 1) & (predicted[3] == 1)
    
    if (j + 1) % 10 == 0:
        print('test iteration:', j + 1)

print('accuracy:', accuracy / num_test * 100, '%')






