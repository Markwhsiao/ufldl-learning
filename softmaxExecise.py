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
        return inputSize,numClasses,lmbda

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


def loadTestData():
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
        images = load_MNIST.loadMniImageSet("./mnist/t10k-images-idx3-ubyte")
        labels = load_MNIST.loadMniLabelSet("./mnist/t10k-labels-idx1-ubyte")
        return images,labels

def softmaxCost(theta, numClasses, inputSize, lmbda, data, labels):
        theta = theta.reshape(numClasses, inputSize)
        numCases = data.shape[1]
        groundTruth = sparse.csr_matrix( (np.ones(numCases), (labels, np.array(range(numCases)))),shape=(numClasses,numCases) )
        groundTruth = np.array(groundTruth.todense())
        ####YOUR CODE HERE
        theta_data = np.dot(theta,data)
        theta_data = theta_data - np.max(theta_data)
        prob = np.exp(theta_data) / np.sum(np.exp(theta_data),axis=0)
        thetagrad = (-1 / numCases) * np.dot(groundTruth - prob,data.T) + lmbda*theta
        cost = - 1/numCases * np.sum(np.log(prob)*groundTruth) + lmbda/2.0 * np.sum(np.square(theta))
        return cost,thetagrad.flatten()

def softmaxTrain(input_size, num_classes, lambda_, data, labels, options={'maxiter': 100, 'disp': True}):
        theta = 0.005 * np.random.rand(num_classes * input_size)
        J = lambda x: softmaxCost(x, num_classes, input_size, lambda_, data, labels)
        result = optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
        opt_theta = result.x
        return opt_theta,input_size,num_classes

def softmaxPredict(opt_theta,testData,testLabels,numClasses,inputSize):
    opt_theta = opt_theta.reshape(numClasses,inputSize)
    exp_theta_data = np.exp(opt_theta.dot(testData))/np.sum(np.exp(opt_theta.dot(testData)), axis=0)
    pred_class = exp_theta_data.argmax(axis=0)
    print "Accuracy: {0:.2f}%".format(100 * np.sum(pred_class == testLabels, dtype=np.float64) / testLabels.shape[0])
    



def check_softmax_cost(numClasses, inputSize, lmbda, images, labels):
    inputData   = images[:,0:30]
    inputLabels = labels[0:30]
    theta = 0.005 * np.random.rand(numClasses * inputSize);
    J = lambda x :softmaxCost(x, numClasses, inputSize, lmbda,inputData, inputLabels)
    check_gradient.check_compute_gradient(J,theta)
     




if __name__ == "__main__":
    inputSize,numClasses,lmbda = init()
    images,labels = loadTrainData()
    np.set_printoptions(threshold='nan')
    #test softmaxCost function 
    check_softmax_cost(numClasses, inputSize, lmbda, images, labels)
    opt_theta,inputSize,numClasses = softmaxTrain(inputSize,numClasses,lmbda,images,labels)
    #predict 
    testImage,testLabels = loadTestData()
    softmaxPredict(opt_theta,testImage,testLabels,numClasses,inputSize)
