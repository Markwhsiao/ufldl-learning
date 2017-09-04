import numpy as np
import load_MNIST
import check_gradient
from numpy.matlib import repmat
import scipy.optimize as optimize

def initialize(hidden_size, visible_size):
    # we'll choose weights uniformly from the interval [-r, r]
    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random((hidden_size, visible_size)) * 2 * r - r
    W2 = np.random.random((visible_size, hidden_size)) * 2 * r - r
    b1 = np.zeros(hidden_size, dtype=np.float64)
    b2 = np.zeros(visible_size, dtype=np.float64)
    theta = np.concatenate((W1.reshape(hidden_size * visible_size),
                            W2.reshape(hidden_size * visible_size),
                            b1.reshape(hidden_size),
                            b2.reshape(visible_size)))
    return theta

def initInputParameters():
    visibleSize = 28*28
    hiddenSize  = 196
    sparsityParam = 0.01
    lambda_ = 0.0001
    beta = 3
    return visibleSize,hiddenSize,sparsityParam,lambda_,beta

def loadImages():
    train_images = load_MNIST.loadMniImageSet("mnist/train-images-idx3-ubyte")
    return train_images

def sparseAutoencoderCost(theta,visibleSize,hiddenSize,lambda_,sparsityParam,beta,data):
    def sigmod(x):
        return 1.0 / (1.0 + np.exp(-x))
    def grad_sigmod(x):
        ans_sigm  = sigmod(x)
        grad_sigm =  ans_sigm - ans_sigm * ans_sigm;
        return grad_sigm

    W1 = theta[0:hiddenSize*visibleSize]
    W1 = W1.reshape((visibleSize,hiddenSize)).transpose()
    W2 = theta[hiddenSize*visibleSize:hiddenSize*visibleSize*2]
    W2 = W2.reshape((hiddenSize,visibleSize)).transpose()
    b1 = theta[hiddenSize*visibleSize*2:hiddenSize*visibleSize*2 + hiddenSize]
    b2 = theta[hiddenSize*visibleSize*2 + hiddenSize:theta.shape[0]]

    cost = 0.0
    y = data
    m = data.shape[1]
    ##forword
    a1 = data
    z2 = W1.dot(data) + repmat(b1,m,1).transpose() 
    a2 = sigmod(z2)
    z3 = W2.dot(a2)  + repmat(b2,m,1).transpose()
    a3 = sigmod(z3)

    rho_p = np.sum(a2,axis=1) / m
    rho = sparsityParam
    kl_rho = rho*np.log(rho/rho_p) + (1-rho)*np.log((1-rho)/(1-rho_p))

    ###backword
    kl_g = -rho / rho_p + (1-rho) / (1-rho_p)
    delta3  = -(y-a3) * grad_sigmod(z3)
    delta2  = ((W2.transpose()).dot(delta3) + beta*repmat(kl_g,m,1).transpose() )* grad_sigmod(z2)
    W1grad  = delta2.dot(a1.transpose()) / m + lambda_*W1
    W2grad  = delta3.dot(a2.transpose()) / m + lambda_*W2
    b1grad  = np.sum(delta2,1)/m
    b2grad  = np.sum(delta3,1)/m
    error_sq = 0.5*(y-a3)*(y-a3)
    cost = 1.0/m *sum(error_sq.flatten()) + lambda_/2.0 * (np.sum(W1**2) + np.sum(W2**2)) + beta*np.sum(kl_rho)
    grad = np.append(W1grad.transpose().flatten(),W2grad.transpose().flatten())
    grad = np.append(grad,b1grad.flatten())
    grad = np.append(grad,b2grad.flatten())
    return cost,grad

def check_sparseAutoencoderCost(data):
    data = data[0:16,0:10]
    visibleSize = 4*4
    hiddenSize  = 9
    sparsityParam = 0.01
    lambda_ = 0.0001
    beta = 3
    theta = initialize(9,16)
    J = lambda x : sparseAutoencoderCost(x,visibleSize,hiddenSize,lambda_,sparsityParam,beta,data)
    check_gradient.check_compute_gradient(J,theta)



def softmaxTrain(visibleSize,hiddenSize,lambda_,sparsityParam,beta,data, options={'maxiter': 400, 'disp': True}):
    theta = initialize(hiddenSize,visibleSize)
    J = lambda x : sparseAutoencoderCost(x,visibleSize,hiddenSize,lambda_,sparsityParam,beta,data)
    result = optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
    opt_theta = result.x
    return opt_theta,input_size,num_classes

if __name__ == "__main__":
    data = loadImages()
    check_sparseAutoencoderCost(data)
    visibleSize,hiddenSize,sparsityParam,lambda_,beta = initInputParameters()
    softmaxTrain(visibleSize,hiddenSize,lambda_,sparsityParam,beta,data)    







