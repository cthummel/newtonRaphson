#import a few libraries for use later
import numpy as np
import sys, getopt, math

#Read in data set.
def read(file):
    data = []
    with open(file, encoding="utf8") as f:
        for line in f:
            data.append(float(line))

    return np.array(data)

#g(beta) used in iteration algorithm
def betaFunction(data, beta):
    result = 0
    denom = 0
    for value in data:
        result += np.power(value, beta) * np.log(value)
        denom += np.power(value, beta)
    
    result = result / denom

    result -= 1.0 / beta

    result -= sum(np.log(data)) / len(data)

    #print("betaFunction:", result)

    return result

#g'(beta) used in iteration algorithm
def betaFunctionPrime(data, beta):
    a = 1.0 / np.power(beta,2)

    bTop = 0
    bBottom = 0 
    for value in data:
        bTop += np.power(value, beta) * np.log(value) * np.log(value)
        bBottom += np.power(value, beta)

    b = bTop / bBottom

    cTop = 0
    cBottom = 0
    for value in data:
        cTop += np.power(value, beta) * np.log(value)
        cBottom += np.power(value, beta)

    c = np.power(cTop, 2) / np.power(cBottom, 2)

    #print("betaFunctionPrime:",  a+b-c)

    return a + b - c



def betaSecondDerivative(data, beta, theta):
    a = -len(data) / np.power(beta, 2)

    b = 0
    for value in data:
        b += np.power(value / theta, beta) * np.power(np.log(value / theta), 2)

    #print(a - b)

    return a - b

#Observed theta variance
def thetaSecondDerivative(data, beta, theta):
    a = len(data) * beta / np.power(theta, 2)

    b = 0
    for value in data:
        b += np.power(value / theta, beta)

    b = b * (np.power(beta, 2) + beta) / np.power(theta, 2)

    return a - b


#Observered Covariance
#dl / dtheta dbeta = -n/theta + sum((x_i / theta)^beta * (1 + (beta * ln(x_i / theta))) / theta
def mixedSecondDerivative(data, beta, theta):
    a = -len(data) / theta

    b = 0
    for value in data:
        b += np.power(value / theta, beta) * (1 + (beta * np.log(value / theta)))

    b = b / theta

    #print(a, b, a+b)

    return a + b

#Given a bHat, we can get the thetaHat
def thetaFunction(data, beta):
    return np.power(np.sum(np.power(data, beta)) / len(data), 1.0 / beta)


#Generate the information matrix
def informationMatrix(data, beta, theta):
    matrix = np.array([[0.0,0.0], [0.0,0.0]])
    matrix[0,0] = -betaSecondDerivative(data, beta, theta)
    matrix[0,1] = matrix[1,0] = -mixedSecondDerivative(data, beta, theta)
    matrix[1,1] = -thetaSecondDerivative(data, beta, theta)

    return matrix



#Iteration algorithm. Only works for Weibull parameter estimation.
def newtonRaphsonIteration(data, betaStart):
    betaHat = betaStart
    updateStepValue = 1
    iterationCount = 0
    while abs(updateStepValue) > .0001:
        #print("beta:", betaHat)
        #print("Update Step", updateStepValue)

        updateStepValue = (betaFunction(data, betaHat) / betaFunctionPrime(data, betaHat))
        betaHat -= updateStepValue
        
        iterationCount += 1
    
    print("Converged after ", iterationCount, "steps.", "\n")

    theta = thetaFunction(data, betaHat)
    matrix = informationMatrix(data, betaHat, theta)
    matrixInv = np.linalg.inv(matrix)

    #betaHatSE = np.sqrt(1.0 / (len(data) * matrixInv[0,0]))
    #thetaSE = np.sqrt(1.0 / (len(data) * matrixInv[1,1]))
    #betaHatSE = 1.0 / (np.sqrt(matrixInv[0,0]) * len(data))
    #thetaSE = 1.0 / (np.sqrt(matrixInv[1,1]) * len(data))
    betaHatSE = np.sqrt(matrixInv[0,0])
    thetaSE = np.sqrt(matrixInv[1,1])

    print("Beta:\t", betaHat,"\tSE:", betaHatSE ,"\t95% CI:(", betaHat - (1.96 * betaHatSE), ",", betaHat + (1.96 * betaHatSE), ")")
    print("Theta:\t", theta, "\tSE:", thetaSE, "\t95% CI:(", theta - (1.96 * thetaSE), ",", theta + (1.96 * thetaSE), ")\n")
    print("Observed Information Matrix:", matrix)
    print()
    print("Inverse:", matrixInv)

#Driver method that handles command line arguments and calls the algorithm.
def main(argv):
    opts, args = getopt.getopt(argv, "hi:", ['--input'])
    for opt, arg in opts:
        if opt == '-h':
            print("I need help too.")
            sys.exit(1)
        elif opt in ('-i', '--input'):
            inputFile = arg


    dataset = read(inputFile)
    newtonRaphsonIteration(dataset, .1)




if __name__ == '__main__':
    main(sys.argv[1:])  