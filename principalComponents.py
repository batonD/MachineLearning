import numpy
 
def principalComponents(data):

 	# make sure data has zero mean
 	# calculate covariance matrix
    dev = (data.T - numpy.mean(data, axis=1)).T
    cov = numpy.cov(dev)
    eigenvalues, pc = numpy.linalg.eig(cov)
 
    # sort eigenvalues in decreasing order
    idxList = numpy.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idxList]
    pc = pc[:, idxList]
 
    return eigenvalues, pc