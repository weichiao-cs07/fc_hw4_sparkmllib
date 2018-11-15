import sys
import pandas as pd
import numpy as np
import math 
from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import RegressionMetrics

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)    
    
def SetPath(sc):
    global Path
    if sc.master[0:5]=="local" :
        Path="file:/home/u0756527/"
    else:   
        return 0

def extract_label(record):
    label=(record[-1])
    return float(label)

def convert_float(x):
    return (0 if x=="?" else float(x))

def extract_features(record,featureEnd):
    featureSeason=[convert_float(field)  for  field in record[2]] 
    features=[convert_float(field)  for  field in record[4: featureEnd-2]]
    return  np.concatenate( (featureSeason, features))

def extract_features_count(record,featureEnd):
    featureSeason=[convert_float(field)  for  field in record[2]] 
    features=[convert_float(field)  for  field in record[4: featureEnd-2]]
    featureCount = [convert_float(field)  for  field in record[16]] 
    return  np.concatenate( (featureSeason, features, featureCount))


def PrepareData(sc): 
    rawDataWithHeader = sc.textFile(Path+"Downloads/dataset/hour.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)
    lines = rawData.map(lambda x: x.split(","))
    labelpointRDD = lines.map(lambda r:LabeledPoint(extract_label(r), extract_features(r,len(r) - 1)))
    (trainData, validationData) = labelpointRDD.randomSplit([7, 3])
    return (trainData, validationData)


        
def trainEvaluateModel(trainData,validationData,impurityParm, maxDepthParm, maxBinsParm):
    model = DecisionTree.trainRegressor(trainData,categoricalFeaturesInfo={},impurity=impurityParm,maxDepth=maxDepthParm,maxBins=maxBinsParm)
    return model

def PredictTestData(model): 
    rawDataWithHeader = sc.textFile(Path+"Downloads/dataset/hour.csv")
    header = rawDataWithHeader.first() 
    rawData = rawDataWithHeader.filter(lambda x:x !=header)    
    lines = rawData.map(lambda x: x.split(","))
    (trainLine, validLine) = lines.randomSplit([7, 3])
    trainData = trainLine.map(lambda r:LabeledPoint(extract_label(r), extract_features(r,len(r) - 1)))
    validationData = validLine.map(lambda r:LabeledPoint(extract_label(r), extract_features(r,len(r) - 1)))
    dataRDD = validLine.map(lambda r: ( r[0] ,extract_features_count(r,len(r))))
    for data in dataRDD.take(10):
        predictResult = model.predict(data[1])
        print(" season：" +str(data[1][1]) +" mnth："+str(data[1][2])+" hr：" +str(data[1][3]) +" holiday：" +str(data[1][4])\
              +" week："+str(data[1][5])+" workday："+str(data[1][6])+" weathersit："+str(data[1][7])+" temp："+str(data[1][8])\
              +" atemp："+str(data[1][9])+" hum："+str(data[1][10])+" windspeed："+str(data[1][11])+" cnt："+str(data[1][12])+\
              " predict："+str(predictResult))



def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels=score.zip(validationData.map(lambda p: p.label))
    metrics = RegressionMetrics(scoreAndLabels)
    RMSE=metrics.rootMeanSquaredError
    return(RMSE)

def CreateSparkContext():
    sparkConf = SparkConf().setAppName("RunDecisionTreeRegression").set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    SetLogger(sc)
    SetPath(sc)
    return (sc)



if __name__ == "__main__":
    sc=CreateSparkContext()
    print("read data")
    (trainData, validationData) =PrepareData(sc)
    trainData.persist(); validationData.persist(); 
    print("train model")
    model= trainEvaluateModel(trainData, validationData, "variance", 10, 200)
    print("predict")
    PredictTestData(model)
    print("eval model")    
    AUC = evaluateModel(model, validationData)
    print(AUC)
    