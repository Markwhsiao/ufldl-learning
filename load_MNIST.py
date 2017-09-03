import numpy as np

def loadMniImageSet(filename):
    f = open(filename,"r")
    magic    = np.fromfile(f,dtype=np.dtype('>i4'),count=1)
    imageNum = np.fromfile(f,dtype=np.dtype('>i4'),count=1)
    width    = np.fromfile(f,dtype=np.dtype('>i4'),count=1)
    hght     = np.fromfile(f,dtype=np.dtype('>i4'),count=1)
    
    images   = np.fromfile(f,dtype=np.ubyte)
    images   = images.reshape((imageNum,width*hght)).transpose()
    images   = images.astype(np.float64)/255
    f.close()
    return images

def loadMniLabelSet(filename):
    f = open(filename,"r")
    magic    = np.fromfile(f,dtype='>i4',count=1)
    imageNum = np.fromfile(f,dtype='>i4',count=1)
    labels = np.fromfile(f,dtype=np.ubyte)
    f.close()
    return labels

if __name__ == "__main__":
    train_image  = loadMniImageSet("./mnist/train-images-idx3-ubyte")
    train_labels = loadMniLabelSet("./mnist/train-labels-idx1-ubyte")
    test_image   = loadMniImageSet("./mnist/t10k-images-idx3-ubyte")
    test_label   = loadMniLabelSet("./mnist/t10k-labels-idx1-ubyte")
    np.savetxt("train_image.txt",train_image)
    np.savetxt("train_labels.txt",train_labels)
    np.savetxt("test_image.txt",test_image)
    np.savetxt("test_label.txt",test_label)
