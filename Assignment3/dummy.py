import sys
import os
import matplotlib.pyplot as plt
import numpy as np
 
class nn_linear_layer:
    
    # linear layer.
    # randomly initialized by creating matrix W and bias b
    def __init__(self, input_size, output_size, std=1):
        self.W = np.random.normal(0,std,(output_size,input_size))
        print(f"W is set as {self.W}")
        self.b = np.random.normal(0,std,(output_size,1))
        print(f"b is set as {self.b}")
    
    ######
    ## Q1
    def forward(self,x):
        x_3d = x[:,:,np.newaxis]
        # x is now 20,2,1
        b_transpose = np.reshape(self.b,(1,-1)) # row vector (w. 2axis. that is (1 x ?))
        res = np.matmul(self.W, x_3d).squeeze(-1) + b_transpose
        # 검증 완료
        # res is  20,4
        return res
    
   
    ######
    ## Q2
    ## returns three parameters
    def backprop(self,x,dLdy):
        dydx = self.W
        dydb = np.identity(len(self.W))
        dydW = x.mean(axis=0)
        
        dLdb = np.matmul(np.reshape(dLdy,(1,-1)),dydb)
        dLdx = np.matmul(np.reshape(dLdy,(1,-1)),dydx)
        dLdW = np.matmul(np.reshape(dLdy,(-1,1)),np.reshape(dydW,(1,-1)))


        """print(f"dLdb shape: {np.shape(dLdb)}")
        print(f"dLdx shape: {np.shape(dLdx)}")
        print(f"dLdW shape: {np.shape(dLdW)}")"""
        # dLdy -> (batch size, output size) ex) layer2 : (20,2)
        # x -> (batch size, input size) ex) layer2 : (20,4)
        # W -> (output size, input size) ex) layer2 : (2,4)
        # x[i] means one of input vector which is 'row' vector.
        """dLdW = []
        #dLdb = []
        dLdx = []
        for i in range(0,len(x)):
            #dydb = np.identity(out_size)
            # because dydb is identical matrix, so we don't have to run matmul
            dydx = self.W
            dydW = x[i]            
            row = dLdy[i]
            #dLdb.append(np.matmul(row,dydb))
            dLdx.append(np.matmul(np.reshape(row,(1,-1)),dydx)[0])
            dLdW.append(np.matmul(np.reshape(row,(-1,1)),np.reshape(dydW,(1,-1))))

        dLdW = np.array(dLdW)
        dLdW = dLdW.mean(axis = 0)
        dLdx = np.array(dLdx)
        dLdb = dLdy.mean(axis = 0)[:,np.newaxis].T
        # dLdb 는 dLdy의 각 행에 identity matrix 곱한후 열방향 평균낸것.
        #print(f"dLdb : {dLdb}")"""
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
        #res = np.array([ 1/ (1 + np.exp(-t)) for t in x])
        res = np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
        # 검증 완료
        #
        return res
    
    ######
    ## Q4
    def backprop(self,x,dLdy):
        """x_mean = x.mean(axis=0)
        dLds1 = []
        dldy_minonedimension = dLdy[0]
        for i in range(len(x_mean)):
            t = x_mean[i]
            # overflow 방지
            t= np.clip(t,-300,300)
            sigmoid = 1 / (1 + np.exp(t))
            derivative = (1 - sigmoid) * sigmoid
            dLds1.append(derivative * dldy_minonedimension[i])
        dLds1 = np.array(dLds1)
        dLds1 = np.reshape(dLds1,(1,-1))"""
        dLds1 = []
        for i in range(0,len(dLdy)):
            row = []
            for j in range(0,len(dLdy[i])):
                input = np.clip(x[i][j],-500,500)
                if input >= 0:
                    sigmoid = 1 / (1 + np.exp(-input))
                else:
                    sigmoid = np.exp(input) / (1 + np.exp(input))
                derivative = (1 - sigmoid) * (sigmoid)
                row.append(dLdy[i][j] * derivative)
            dLds1.append(np.array(row))
            #dyds1.append(np.array([np.exp(-t) / ((1 + np.exp(-t))**2) for t in dLdy[i]]))
        dLds1 = np.array(dLds1).mean(axis=0)
        dLds1 = np.reshape(dLds1,(-1,1))

        """x_mean = x.mean(axis=0) # (4,)
        dLds1 = []
        for i in range(len(x)):
            row = []
            dldy_minonedimension = dLdy[i]
            for j in range(len(x[i])):
                t = x_mean[i][j]
                # overflow 방지
                if t >= 0:
                    sigmoid = 1 / (1 + np.exp(-t))
                else:
                    sigmoid = np.exp(t) / (1 + np.exp(t))
                derivative = (1 - sigmoid) * sigmoid
                row.append(derivative * dldy_minonedimension[i])
            dLds1.append(row)
        dLds1 = np.array(dLds1)
        dLds1 = np.reshape(dLds1,(1,-1))"""
        """dLds1 = []
        for i in range(0,len(dLdy)):
            row = []
            for j in range(0,len(dLdy[i])):
                input = np.clip(x[i][j],-500,500)
                if input >= 0:
                    sigmoid = 1 / (1 + np.exp(-input))
                else:
                    sigmoid = np.exp(input) / (1 + np.exp(input))
                derivative = (1 - sigmoid) * (sigmoid)
                row.append(dLdy[i][j] * derivative)
            dLds1.append(np.array(row))
            #dyds1.append(np.array([np.exp(-t) / ((1 + np.exp(-t))**2) for t in dLdy[i]]))
        dLds1 = np.array(dLds1).mean(axis=0)
        dLds1 = np.reshape(dLds1,(-1,1))"""
        return dLds1


class nn_softmax_layer:
    def __init__(self):
        pass
    ######
    ## Q5
    def forward(self,x):
        res = []
        for i in range(0,len(x)):
            denominator = 0
            row = []
            x_max = np.max(x[i])
            # 자꾸 오버플로우가 발생해서, max값을 빼주었음.
            for j in range(0,len(x[i])):
                denominator += np.exp(x[i][j]-x_max)
            for j in range(0,len(x[i])):
                row.append(np.exp(x[i][j]-x_max) / denominator)
            res.append(row)
        res = np.array(res)
        # 검증완료
        return res
    
    ######
    ## Q6
    def backprop(self,x,dLdp):
        # dLdp -> 2차원 row vector   (1,2)
        x_mean = x.mean(axis=0)
        dpds2 = np.array( [[ x_mean[0]*(1-x_mean[0]), -x_mean[0]*x_mean[1] ], \
                          [-x_mean[0]*x_mean[1] , x_mean[1]*(1-x_mean[1])]])
        dLds2 = np.matmul(dLdp,dpds2)
        """dLds2 = []
        for i in range(0,len(x)):
            pre = np.array(dLdp[i])
            dpds2 = np.array([[pre[0] * (1 - pre[0]) , -pre[0]*pre[1]], \
                    [-pre[0]*pre[1]  , pre[1] * (1 - pre[1])]])
            pre = np.reshape(np.array(dLdp[i]),(1,-1))
            dLds2.append(np.matmul(pre,dpds2).squeeze())
        dLds2 = np.array(dLds2).mean(axis=0)
        dLds2 = np.reshape(dLds2,(-1,1))"""
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
            c_e = 0
            for j in range(0,len(x[i])):
                p_i = x[i][j]
                p_i = np.clip(p_i, 1e-15, 1 - 1e-15)
                c_e += b_i * np.log(p_i) + (1-b_i) * np.log(1-p_i)
            cross_entropy.append([-c_e /2])
        cross_entropy = np.array(cross_entropy).mean()
        # 검증 완료
        return cross_entropy
        
    ######
    ## Q8
    def backprop(self,x,y):
        dLdp = []
        for i in range(0,len(x)):
            row = []
            b = y[i][0]
            for j in range(0,len(x[i])):
                p_i = x[i][j]
                p_i = np.clip(p_i, 1e-15, 1 - 1e-15)
                grad_L = (1-b) / (1-p_i) - b / p_i
                row.append(grad_L/2)
            dLdp.append(np.array(row))
        dLdp = np.array(dLdp).mean(axis = 0)
        dLdp = np.reshape(dLdp,(1,-1))
        # dLdp -> (2,1)
        # 혼동 방지를 위해 row vector로 통일 -> Jacobian!
        # [dLdp_1, dLdp_2] * 20times, that is, [grad_p_L] * 20times then, get mean
        # 검증 완료
        return dLdp

# number of data points for each of (0,0), (0,1), (1,0) and (1,1)
num_d=5

# number of test runs
num_test=40

## Q9. Hyperparameter setting
## learning rate (lr)and number of gradient descent steps (num_gd_step)
## This part is not graded (there is no definitive answer).
## You can set this hyperparameters through experiments.
lr= 0.0001
num_gd_step= 100000

# dataset size
batch_size=4*num_d

# number of classes is 2
num_class=2

# variable to measure accuracy
accuracy=0

# set this True if want to plot training data
show_train_data=True

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
    print(f"l1_out : {(l1_out)}")  #---------- okay
    # activation
    a1_out = act.forward(l1_out)
    print(f"a1_out : {(a1_out)}")  # ---------- okay
    
    # output layer
    # linear
    l2_out = layer2.forward(a1_out)
    print(f"l2_out : {(l2_out)}")  #---------- okay

    # softmax
    smax_out = smax.forward(l2_out)
    print(f"smax_out : {(smax_out)}") #---------- okay

    # cross entropy loss
    loss_out[i] = cent.forward(smax_out, y_train)
    print(f"loss_out : {loss_out[i]}") #------------ okay
    
    """if i == 2:
        if j == 1:
            print('error')
        print("error")"""
    ################
    # perform backprop
    # output layer
    # cross entropy
    b_cent_out = cent.backprop(smax_out, y_train)
    #print(f"b_cent_out : {(b_cent_out)}") --------- okay
    # softmax
    b_nce_smax_out = smax.backprop(l2_out, b_cent_out)
    #print(f"b_nce_smax_out : {(b_nce_smax_out)}") --------- okay
    # linear
    b_dLdW_2, b_dLdb_2, b_dLdx_2 = layer2.backprop(x=a1_out, dLdy=b_nce_smax_out)
    """print(f"b_dLdW_2 : {(b_dLdW_2)}")
    print(f"b_dLdb_2 : {(b_dLdb_2)}")
    print(f"b_dLdx_2 : {(b_dLdx_2)}")""" #  ------ okay
    # backprop, hidden layer
    # activation
    b_act_out = act.backprop(x=l1_out, dLdy=b_dLdx_2)
    #print(f"b_act_out : {(b_act_out)}") # ------- nan occurs
    # linear
    b_dLdW_1, b_dLdb_1, b_dLdx_1 = layer1.backprop(x=x_train, dLdy=b_act_out)
    #print(f"b_dLdW_1 : {np.shape(b_dLdW_1)}")
    #print(f"b_dLdb_1 : {np.shape(b_dLdb_1)}")
    #print(f"b_dLdx_1 : {np.shape(b_dLdx_1)}")
    
    ################
    # update weights: perform gradient descent
    layer2.update_weights(dLdW=-b_dLdW_2 * lr, dLdb=-b_dLdb_2.T * lr)
    layer1.update_weights(dLdW=-b_dLdW_1 * lr, dLdb=-b_dLdb_1.T * lr)
    
    #Debug!
    if layer2.W[0][0] == np.nan:
        print(f"Error at {i}th iteration")
        break

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






