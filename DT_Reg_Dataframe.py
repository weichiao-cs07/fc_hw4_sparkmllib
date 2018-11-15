import pyspark.sql.types 
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf,col
from pyspark.sql import SQLContext
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer,VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


global Path    
Path="file:/home/u0756527/"
def CreateSparkContext():
    def SetLogger( sc ):
        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
        logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
        logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)
    sparkConf = SparkConf().setAppName("RunDecisionTreeBinary").set("spark.ui.showConsoleProgress", "false") 
    sc = SparkContext(conf = sparkConf)
    SetLogger(sc)
    return (sc)

sc=CreateSparkContext()
sqlContext = SQLContext(sc)
row_df = sqlContext.read.format("csv").option("header", "true").load(Path+"Downloads/dataset/hour.csv")

print("read data")
new_row_df=row_df.drop("instant").drop("dteday").drop('yr').drop("casual").drop("registered")
new_df= new_row_df.select([ col(column).cast("double").alias(column) for column in new_row_df.columns])
train_df, test_df = new_df.randomSplit([0.7, 0.3])
train_df.cache()
test_df.cache()

assemblerInputs = new_df.columns[:-1]
print("setup pipeline")
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="tmp_features")
indexer = VectorIndexer(inputCol="tmp_features", outputCol="features", maxCategories=24)
dt = DecisionTreeRegressor(labelCol="cnt",featuresCol= "features",maxDepth=10, maxBins=100,impurity="variance")
dt_pipeline = Pipeline(stages=[assembler,indexer ,dt])

print("train model")
dt_pipelineModel = dt_pipeline.fit(train_df)

print("predict")
predicted=dt_pipelineModel.transform(test_df).select("season","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","cnt","prediction").show(10)
print(predicted)

print("eval model")
evaluator = RegressionEvaluator(labelCol='cnt',predictionCol='prediction',metricName="rmse")
predicted_df=dt_pipelineModel.transform(test_df)
rmse = evaluator.evaluate(predicted_df)
print(rmse)
