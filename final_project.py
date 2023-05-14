import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as cluster
import sklearn.mixture as mixture
import seaborn as sb
from yellowbrick.cluster import KElbowVisualizer
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA


def readData(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            cline = lines[i].strip()
            lines[i] = cline
    f.close()
    return lines


def getBlocks(inputLines):
    splitIndices = []
    for i in range(len(inputLines)):
        if (inputLines[i] == ""):
            splitIndices.append(i)
    blocks = []
    for i in range(len(splitIndices) - 1):
        newBlock = inputLines[(splitIndices[i] + 1):(splitIndices[i + 1])]
        blocks.append(newBlock)
    blocks.append(inputLines[splitIndices[(len(splitIndices) - 1)] + 1:])
    return blocks


def splitBlocksByDigit(blocks, blocksPerDigit):
    numDigits = int(len(blocks) / blocksPerDigit)
    start = 0
    digitBlocks = []
    for i in range(numDigits):
        end = start + blocksPerDigit
        digitBlocks.append(blocks[start:end])
        start = end
    return digitBlocks


"""Compute a time series for each MFCC across a block"""


def getTimeSeries(block):
    for i in range(len(block)):
        block[i] = block[i].split(" ")
    finalBlock = np.array(block)
    finalBlock = finalBlock.T
    return finalBlock.astype(float)


def getDigit(blocknum, digitsPerBlock):
    return int(blocknum / digitsPerBlock)


def plotBlock(block, blocknum, digitVal):
    coeffSeries = getTimeSeries(block)
    timeValues = np.arange(0, len(coeffSeries[0]), 1)
    matplotlib.use('TkAgg')
    upperBound = np.max(np.array(coeffSeries))
    lowerBound = np.min(np.array(coeffSeries))
    for i in range(len(coeffSeries)):
        cLabel = "MFCC" + str(i + 1)
        plt.plot(timeValues, coeffSeries[i], label=cLabel)
    plt.ylim([lowerBound, upperBound])
    plt.title("MFCC Time Series for Block " + str(blocknum) + " (Digit: " + str(digitVal) + ")")
    plt.xlabel("Analysis Window Index")
    plt.ylabel("MFCC Value")
    plt.grid()
    plt.legend()
    plt.show()


trainingLines = readData('data/Train_Arabic_Digit.txt')
testLines = readData('data/Test_Arabic_Digit.txt')

# A block is a distinct utterance of a single digit
# 66 Speakers, 10 iterations per digit, 10 digits, 6600 total blocks, 660 per digit
trainingBlocks = getBlocks(trainingLines)
testBlocks = getBlocks(testLines)
# Index corresponds to the digit value and yields all its blocks
trainingBlocksByDigit = splitBlocksByDigit(trainingBlocks, 660)
testBlocksByDigit = splitBlocksByDigit(testBlocks, 220)


# testBlock = 1300
# plotBlock(trainingBlocks[testBlock], testBlock, getDigit(testBlock, 660))

# 660 spoken block, 40 frames per block
def concatFrames(dataInput):
    digitToCoeffPointsMap = {}
    for i in range(0, 10):
        digitData = []
        for frame in dataInput[i]:
            for strframe in frame:
                floats = [float(x) for x in strframe.split()]
                digitData.append(floats)
        digitToCoeffPointsMap[i] = digitData
    return digitToCoeffPointsMap


trainMapping = concatFrames(trainingBlocksByDigit)
testMapping = concatFrames(testBlocksByDigit)


def getTupleFromKMeans(kMeansRes, data):
    clusterRange = kMeansRes.n_clusters
    clusters = {}
    for i in range(0, clusterRange):
        clusters[i] = []
    for i in range(len(kMeansRes.labels_)):
        clusters[kMeansRes.labels_[i]].append(data[i])
    tuples = []
    for cluster in clusters.keys():
        currVals = clusters[cluster]
        mean = np.array(currVals).mean(axis=0)
        covar = np.cov(np.array(currVals).T)
        weight = (len(currVals) / len(data))
        currTup = []
        currTup.append(mean)
        currTup.append(covar)
        currTup.append(weight)
        tuples.append(currTup)
    return tuples


def kMeansGMM(dataVals):
    clusterCenterNumforDigit = {0: 4, 1: 4, 2: 3, 3: 3, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4}
    kMeansResults = {}
    for digit in clusterCenterNumforDigit.keys():
        kMeansResults[digit] = cluster.KMeans(n_clusters=clusterCenterNumforDigit[digit])
        kMeansResults[digit].fit(dataVals[digit])
    digitToGMMComponentTuples = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for digit in kMeansResults:
        digitToGMMComponentTuples[digit] = getTupleFromKMeans(kMeansResults[digit], dataVals[digit])
    return digitToGMMComponentTuples

def emGMM(dataVals):
    clusterNumforDigit = {0: 4, 1: 4, 2: 3, 3: 3, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4}
    digitToGMMComponentTuples = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for key in dataVals.keys():
        for i in range(len(dataVals[key])):
            point = dataVals[key][i][1:]
            dataVals[key][i] = point
    for digit in clusterNumforDigit:
        GMMmodel = mixture.GaussianMixture(n_components=clusterNumforDigit[digit])
        GMMmodel.fit(dataVals[digit])
        newTup = []
        for i in range(len(GMMmodel.weights_)):
            addThis = []
            addThis.append(GMMmodel.means_[i])
            addThis.append(GMMmodel.covariances_[i])
            addThis.append(GMMmodel.weights_[i])
            newTup.append(addThis)
        digitToGMMComponentTuples[digit] = newTup
    return digitToGMMComponentTuples


def maxLikelihoodClassifier(testBlocks, GMMs):
    correctNum = 0
    total = 0
    # correct at 0, prediction at 1
    correctLabels = []
    predictions = []
    accuracy = 0
    # loop thru frames and find liklihood array of each frame, keep multiplying out
    for i in range(len(testBlocks)):
        correctLabel = getDigit(i, 220)
        likelihoods = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 0: 1}
        for frame in testBlocks[i]:
            frame = [float(x) for x in frame.split()]
            frame = frame[1:]
            for digit in GMMs.keys():
                likelihoods[digit] *= evalGMM(frame, GMMs[digit])
        predictedLabel = max(likelihoods, key=likelihoods.get)
        correctLabels.append(correctLabel)
        predictions.append(predictedLabel)
        total += 1
        if (predictedLabel == correctLabel):
            correctNum += 1
        accuracy = correctNum / total
        print(accuracy)
        print(str(i / len(testBlocks)) + " done")
        print(str(predictedLabel) + " last classification")
    return correctLabels, predictions


def evalGMM(point, GMM):
    likelihood = 0
    for component in GMM:
        rv = multivariate_normal(component[0], component[1])
        likelihood += component[2] * rv.pdf(point)
    return likelihood


def plotMatrix(trueVals, dlabels, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(trueVals, dlabels, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    sb.heatmap(cm, cmap="Greens", annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(title+".jpg", bbox_inches='tight')
    plt.show()

#digitKMeansGMMs = kMeansGMM(trainMapping)
#datalabels, classifications = maxLikelihoodClassifier(testBlocks, digitKMeansGMMs)
#plotMatrix(datalabels, classifications, "K-Means Generated GMM's ML Classification Result")


emGMMs = emGMM(trainMapping)
datalabels, classifications = maxLikelihoodClassifier(testBlocks, emGMMs)
plotMatrix(datalabels, classifications, "Final Model")
