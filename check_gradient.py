import numpy as np


def simple_quadratic_function(x):
   value = x[0] ** 2 + 3 * x[0] *x[1]
   grad = np.zeros(shape=2,dtype=np.float32)
   grad[0] = 2*x[0]  + 3*x[1]
   grad[1] = 3*x[0]
   return value,grad

def num_compute_gradient(J,theta):
    '''very slowly,can only use to check gradient
    '''
    epsilon = 0.0000001
    theta_e0 = np.array(theta,dtype = np.float64)
    theta_e1 = np.array(theta,dtype = np.float64)
    grad = np.zeros(theta.shape,dtype = np.float64)

    for i in range(theta.shape[0]):
        theta_e0[i] -= epsilon
        theta_e1[i] += epsilon
        grad[i] = (J(theta_e1)[0] - J(theta_e0)[0])/(2*epsilon)
        theta_e0[i] += epsilon
        theta_e1[i] -= epsilon
    return grad

def check_compute_gradient(J,theta):
    (value,grad) = J(theta)
    num_grad = num_compute_gradient(J,theta)
    print type(num_grad)
    print type(grad)
    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    if diff < 1e-9:
        print "check Success"
        return True
    else:
        print np.array([[num_grad,grad]]).T
        print diff
        return False


if __name__ == "__main__":
    theta = np.array([3,2])
    check_compute_gradient(simple_quadratic_function,theta)
     

         

