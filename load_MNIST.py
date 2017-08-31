import numpy as np
import struct

def reshape(mat,n,m):
        '''
        reshape col main 
        '''
        mat = np.reshape(mat,(n,m),order='F')
        return mat

def loadMniImageSet(filename):
    """
    return [784*m] mat
    """
    print "load image set",filename
    binfile= open(filename, 'rb')
    buffers = binfile.read()
 
    head = struct.unpack_from('>IIII' , buffers ,0)
    print "head,",head
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B' #like '>47040000B'
    imgs = struct.unpack_from(bitsString,buffers,offset)
 
    binfile.close()
    imgs = reshape(imgs,width*height,imgNum)
    print "load imgs finished"
    return imgs

def loadMniLabelSet(filename):
    """
    return [1*m]
    """
    print "load label set",filename
    binfile = open(filename, 'rb')
    buffers = binfile.read()
 
    head = struct.unpack_from('>II' , buffers ,0)
    print "head,",head
    imgNum=head[1]
    offset = struct.calcsize('>II')
    numString = '>'+str(imgNum)+"B"
    labels = struct.unpack_from(numString , buffers , offset)
    binfile.close()
    labels = np.reshape(labels,(imgNum),order='F')
    print 'load label finished'

    return labels


if __name__ == "__main__":
    im     = loadMniImageSet("./mnist/t10k-images-idx3-ubyte")
    labels = loadMniLabelSet("./mnist/train-labels-idx1-ubyte")
    print im.shape
    print labels.shape
