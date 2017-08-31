import numpy as np
import load_MNIST
import scipy.optimize as optimize
import scipy.sparse as sparse
from numpy.matlib import repmat
import check_gradient
#### step0 Init constants and parameters
def init():
        '''
        %% STEP 0: Initialise constants and parameters
        %
        %  Here we define and initialise some constants which allow your code
        %  to be used more generally on any arbitrary input. 
        %  We also initialise some parameters used for tuning the model.
        '''
        inputSize = 28 * 28
        numClasses = 10
        lmbda = 1e-4
        theta = 0.005 * np.random.randn(numClasses * inputSize, 1);
        return inputSize,numClasses,lmbda,theta

def loadTrainData():
        '''
        %  In this section, we load the input and output data.
        %  For softmax regression on MNIST pixels, 
        %  the input data is the images, and 
        %  the output data is the labels.
        %

        % Change the filenames if you've saved the files under different names
        % On some platforms, the files might be saved as 
        % train-images.idx3-ubyte / train-labels.idx1-ubyte
        '''
        images = load_MNIST.loadMniImageSet("./mnist/train-images-idx3-ubyte")
        labels = load_MNIST.loadMniLabelSet("./mnist/train-labels-idx1-ubyte")
        return images,labels


def softmaxCost(theta, numClasses, inputSize, lmbda, data, labels):
        theta = tool.reshape(theta, numClasses, inputSize)
        numCases = data.shape[1]
        groundTruth = np.array(sparse.csr_matrix((np.ones(numCases), (labels.flatten(), np.array(range(numCases))))).todense()) 
        cost = 0
        thetagrad = np.zeros((numClasses,inputSize),dtype=float)
        ####YOUR CODE HERE
        theta = theta - np.max(theta)
        prob = np.exp(theta.dot(data)) 
        rp = repmat(np.sum(prob,axis=0,dtype=float),numClasses,1)
        thetagrad = -1/numCases * np.dot(groundTruth - prob/rp,data.T) + lmbda*theta
        tmp_mat = prob/rp
        tmp_mat = np.log(tmp_mat)
        tmp_mat =  tmp_mat*groundTruth
        cost = - 1/numCases * np.sum(np.square(tmp_mat)) + lmbda/2 * np.sum(np.square(theta))
        return cost,thetagrad.flatten()

def softmax_cost(theta, num_classes, input_size, lambda_, data, labels):
    """
    :param theta:
    :param num_classes: the number of classes
    :param input_size: the size N of input vector
    :param lambda_: weight decay parameter
    :param data: the N x M input matrix, where each column corresponds
                 a single test set
    :param labels: an M x 1 matrix containing the labels for the input data
    """
    m = data.shape[1]
    theta = theta.reshape(num_classes, input_size)
    theta_data = theta.dot(data)
    theta_data = theta_data - np.max(theta_data)
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
    indicator = sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
    indicator = np.array(indicator.todense())
    cost = (-1 / m) * np.sum(indicator * np.log(prob_data)) + (lambda_ / 2) * np.sum(theta * theta)

    grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + lambda_ * theta

    return cost, grad.flatten()
        
def softmax_train(input_size, num_classes, lambda_, data, labels, options={'maxiter': 400, 'disp': True}):
    #softmaxTrain Train a softmax model with the given parameters on the given
    # data. Returns softmaxOptTheta, a vector containing the trained parameters
    # for the model.
    #
    # input_size: the size of an input vector x^(i)
    # num_classes: the number of classes
    # lambda_: weight decay parameter
    # input_data: an N by M matrix containing the input data, such that
    #            inputData(:, c) is the cth input
    # labels: M by 1 matrix containing the class labels for the
    #            corresponding inputs. labels(c) is the class label for
    #            the cth input
    # options (optional): options
    #   options.maxIter: number of iterations to train for

    # Initialize theta randomly
        theta = 0.005 * np.random.randn(num_classes * input_size)
        J = lambda x: softmaxCost(x, num_classes, input_size, lambda_, data, labels)
        result = optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)

        print result
        # Return optimum theta, input size & num classes
        opt_theta = result.x
        return opt_theta, input_size, num_classes

def check_softmax_cost(theta, numClasses, inputSize, lmbda, images, labels):
    inputData   = images[:,0:100]
    inputLabels = labels[0:100]
    J = lambda x : softmaxCost(x, numClasses, inputSize, lmbda,inputData, inputLabels)
    return check_gradient.check_compute_gradient(J,theta)


if __name__ == "__main__":
    inputSize,numClasses,lmbda,theta = init()
    images,labels = loadTrainData()
    softmaxCost(theta, numClasses, inputSize, lmbda, images, labels)
    softmax_train(inputSize,numClasses,lmbda,images,labels)
