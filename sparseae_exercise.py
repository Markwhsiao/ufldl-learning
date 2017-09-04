import numpy as np

def initInputParameters():
	visibleSize = 28*28
	hiddenSize  = 196
	sparsityParam = 0.01
	lambda_ = 0.0001
	beta = 3
	theta = initializeParameters(hiddenSize, visibleSize);
	return visibleSize,hiddenSize,sparsityParam,lambda_,beta,theta

def loadImages():
	train_images = load_MNIST.loadMniImages("mnist/train-images-idx3-ubyte")
	return train_images

def sparseAutoencoderCost(theta,visibleSize,hiddenSize,lambda_,sparsityParam,beta,data):
	
	def sigmod(x):
		return 1.0 / (1.0 + np.exp(-x))
	def grad_sigmod(x):
		ans_sigm  = sigmoid(x)
		grad_sigm =  ans_sigm - ans_sigm * ans_sigm;


	W1 = theta[0:hiddenSize*visibleSize]
	W1 = W1.reshape((visibleSize,hiddenSize)).transpose()
	W2 = theta[hiddenSize*visibleSize:hiddenSize*visibleSize*2]
	W2 = W2.reshape((hiddenSize,visibleSize)).transpose()
	b1 = theta[hiddenSize*visibleSize*2:hiddenSize*visibleSize*2 + hiddenSize]
	b2 = theta[hiddenSize*visibleSize*2 + hiddenSize:theta.shape[0]] #有疑问

	cost = 0.0
	y = data
	m = data.shape[1]
	##forword
	a1 = data
	z2 = W1.dot(data) + np.matlib.repmat(b1,1,m) 
	a2 = sigmod(z2)
	z3 = W2.dot(a2)  + np.matlib.repmat(b2,1,m)
	a3 = sigmod(z3)

	rho_p = np.sum(a2,axis=1) / m
	rho = sparsityParam
	kl_rho = rho*np.log(rho/rho_p) + (1-rho)*np.log((1-rho)/(1-rho_p))

	###backword
	kl_g = -rho / rho_p + (1-rho) / (1-rho_p)
	delta3  = -(y-a3) * grad_sigmod(z3)
	delta2  = ((W2.transpose).dot(delta3) + beta*np.matlib.repmat(kl_g,1,m) )* grad_sigmod(z2)
	W1grad  = delta2.dot(a1.transpose()) / m + lambda_*W1
	W2grad  = delta3.dot(a2.transpose()) / m + lambda_*W2
	b1grad  = np.sum(delta2,1)/m
	b2grad  = np.sum(delta3,1)/m
	error_sq = 0.5*(y-a3).(y-a3)
	cost = 1.0/m *sum(error_sq.flatten()) + lambda_/2.0 * (np.sum(W1**2) + np.sum(W2**2)) 
	       + beta*np.sum(kl_rho)
	grad = np.append(W1grad.flatten(),W2grad.flatten())
	grad = np.append(grad,b1grad.flatten())
	grad = np.append(grad,b2grad.flatten())
	return cost,grad










