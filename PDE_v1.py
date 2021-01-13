import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class ActivationFunction(Enum):
    Tanh = 0
    Sigmoid = 1
    Sin = 2
    Cos = 3
    Atan = 4

class ActivationFunctionNeuralNetwork(nn.Module):
    '''Activation function of the neural network
    '''

    def __init__(self, activationFunction=ActivationFunction.Tanh):
        super(ActivationFunctionNeuralNetwork, self).__init__()
        self.activationFunction = activationFunction


    def forward(self, x):
        if self.activationFunction == ActivationFunction.Tanh:
            return F.tanh()

        elif self.activationFunction == ActivationFunction.Sigmoid:
            return F.sigmoid(x)

        elif self.activationFunction == ActivationFunction.Sin:
            return torch.sin(x)

        elif self.activationFunction == ActivationFunction.Cos:
            return torch.cos(x)

        elif self.activationFunction == ActivationFunction.Atan:
            return torch.atan(x)


class NeuralNet(nn.Module):
    ''' Neural Network used as a mapping function.
        Glorot initialisation.
    '''
    def __init__(self, layers, activationFunction=ActivationFunction.Tanh):
        super(NeuralNet, self).__init__()

        self.layers = []

        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i+1]))
            self.layers.append(ActivationFunctionNeuralNetwork(activationFunction))

        self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))
        self.model = nn.Sequential(*self.layers)
        self.model.apply(self._normal_init)

    def _normal_init(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)

    def PartialDerivative(self, x, order=1):
        if order == 1:
            # to define
            skip

        elif order == 2:
            return torch.autograd.functional.hessian(func=self.model, inputs=x, create_graph=True)

        else:
            raise NotImplementedError

    def train(self, feedDict, lossFunction, iterations):
        xInt = torch.from_numpy(feedDict['xInt'])
        xBound = torch.from_numpy(feedDict['xBound'])
        boundaryCondition = torch.from_numpy(feedDict['boundaryCondition'])

        # default parameters to be customed
        optimizer = torch.optim.LBFGS(
            params=self.model.parameters(),
            lr=1,
            max_iter=20,
            max_eval=None,
            tolerance_grad=1e-07,
            tolerance_change=1e-09,
            history_size=100,
            line_search_fn=None
        )

        def closure():
            optimizer.zero_grad()
            loss = lossFunction(xInt, yInt, yBound, boundaryCondition)
            loss.backward()
            return loss

        for epoch in range(iterations):
            yInt = self.model(xInt.float())
            yBound = self.model(xBound.float())
            optimizer.step(closure)
            print(epoch, loss)
        return loss

    def predict(self, feedDict):
        xInt = Variable(torch.from_numpy(feedDict['xInt']))
        xBound = Variable(torch.from_numpy(feedDict['xBound']))
        boundaryCondition = Variable(torch.from_numpy(feedDict['boundaryCondition']))

        outputInt = self.model(xInt.float())
        outputBound = self.model(xBound.float())

        return outputInt, outputBound

class PDENeuralNetwork():
    def __init__(self, domain, network):
        self.network = network
        self.domain = domain
        self.boundaryDomainSize = []
        self.totalBoundaryDomainSize = 0
        for i in range(len(domain)):
            self.boundaryDomainSize.append(1)
            for j in range(len(domain)):
                if j != i:
                    self.boundaryDomainSize[i] = self.boundaryDomainSize[i] * (self.domain[j][1] - self.domain[j][0])
            self.totalBoundaryDomainSize = self.totalBoundaryDomainSize + 2 * self.boundaryDomainSize[i]

    def SampleInteriorX(self, pointCount):
        ''' Sample points from the domain. Eg domain = [(0, 1), (0, 1)]
            This means sampling in every direction pointCount.
        '''
        if pointCount < 1:
            pointCount = 1

        xInt = []
        for i in range(len(self.domain)):
            xInt.append(np.random.uniform(self.domain[i][0], self.domain[i][1], (pointCount, 1)))
        return xInt

    # Sample uniform collocation points on the boundary of the domain
    def SampleBoundaryX(self, pointCount):
        ''' Sample points from the boundaries
        '''
        if pointCount < 2 * len(self.domain):
            pointCount = 2 * len(self.domain)

        xBound = []
        # Iterate over dimensions
        for i in range(len(self.domain)):
            xBound.append(np.empty((0, 1), dtype=np.float64))

            # Iterate over boundaries
            for j in range(len(self.domain)):
                for bound in self.domain[j]:
                    newPoints = max(int(pointCount * self.boundaryDomainSize[j] / self.totalBoundaryDomainSize), 1)
                    if j == i:
                        newX = np.full((newPoints, 1), bound, dtype=np.float64)
                    else:
                        newX = np.random.uniform(self.domain[j][0], self.domain[j][1],
                                                    (newPoints, 1))
                    xBound[i] = np.concatenate((xBound[i], newX))

        return xBound

    def SampleData(self, interiorPointCount, boundaryPointCount):
        feedDict = {}

        xInt = self.SampleInteriorX(interiorPointCount)
        xBound = self.SampleBoundaryX(boundaryPointCount)
        boundaryCondition = self.BoundaryCondition(xBound)

        feedDict['xInt'] = np.array(xInt).reshape(interiorPointCount, len(self.domain))
        feedDict['xBound'] = np.array(xBound).reshape(boundaryPointCount, len(self.domain))
        feedDict['boundaryCondition'] = np.array(boundaryCondition)

        return feedDict

    def Train(self, interiorPointCount, boundaryPointCount, lossWeight, iterations):

        feedDict = self.SampleData(interiorPointCount=interiorPointCount,
                                   boundaryPointCount=boundaryPointCount)

        self.network.train(feedDict, self.defaultLoss, iterations)

    def Predict(self, interiorPointCount, boundaryPointCount, lossWeight):

        feedDict = self.SampleData(interiorPointCount=interiorPointCount,
                                   boundaryPointCount=boundaryPointCount)

        predictionInterior, predictionBound = self.network.predict(feedDict)

        # Analytical Solution
        self.analyticalInterior = self.AnalyticalSolution(feedDict[xInt])
        self.analyticalBound = self.AnalyticalSolution(feedDict[xBound])

        # Compute L2 error (not sure the sum of interior and boundary is what we need...)
        errorInt = np.sqrt(np.sum((self.analyticalInterior - predictionInterior) ** 2))
        errorBound = np.sqrt(np.sum((self.analyticalBound - predictionBound) ** 2))

        return errorInt+errorBound

class LaplaceBase(PDENeuralNetwork):
    def __init__(self, domain, network=None):
        PDENeuralNetwork.__init__(self, domain, network)

    def defaultLoss(self, xInt, yInt, yBound, boundaryCondition):

        lossInt, lossBound = self.ComputeLossTerms(self.domain, xInt, yInt, yBound, boundaryCondition)

        return torch.add(self.lossWeight*lossInt, (1-self.lossWeight)*lossBound)

    def ComputeLossTerms(self, domain, xInt, yInt, yBound, boundaryCondition):

        # for Laplace operator, take the trace of the hessian
        gradients = [torch.trace(self.network.PartialDerivative(xInt[i,:].float(), order=2))
                     for i in range(xInt.shape[0])]
        lossInt = torch.mean(torch.square(torch.FloatTensor(gradients))) #this may be wrong for computational graph
        lossBound = torch.mean(torch.square(yBound - boundaryCondition))

        return lossInt, lossBound

class Laplace_2d(LaplaceBase):
    def __init__(self, frequency, lossWeight, network=None):
        domain = [(0, 1), (0, 1)]
        LaplaceBase.__init__(self, domain, network)
        self.frequency = frequency
        self.lossWeight = lossWeight

    def AnalyticalSolution(self, x):
        return tf.exp(-x[0] * self.frequency) * tf.sin(x[1] * self.frequency)

    def BoundaryCondition(self, x):
        return np.exp(-x[0] * self.frequency) * np.sin(x[1] * self.frequency)

if __name__=='__main__':

    # Create Neural Network (format is input size, hidden layer size, ..., hidden layer size, output)
    network = NeuralNet([2, 10, 1], ActivationFunction.Sigmoid)

    # Define parameters
    interiorPointCount = 100
    boundaryPointCount = 100
    lossWeight = 0.5
    iterations = 20000

    # Create PDE
    laplace = Laplace_2d(frequency=6 * np.pi, lossWeight = lossWeight, network=network)
    data = laplace.SampleData(100,100)
    xInt = torch.from_numpy(data['xInt']).float()

    laplace.Train(interiorPointCount, boundaryPointCount, lossWeight, iterations)
    error = laplace.Predict(interiorPointCount, boundaryPointCount, lossWeight)
    print(error)
