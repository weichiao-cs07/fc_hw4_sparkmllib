import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.storage.StorageLevel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoderEstimator,VectorAssembler, VectorIndexer}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.RegressionEvaluator
object RunDecisionTreeB {
	case class Data(instant: Int,
			dteday: String,
			season: Double,
			yr: Double,
			mnth: Double,
			hr: Double,
			holiday: Double,
			weekday: Double,
			workingday: Double,
			weathersit: Double,
			temp: Double,
			atemp: Double,
			hum: Double,
			windspeed: Double,
			casual: Double,
			registered: Double,
			cnt: Double)
	case class Data2(
			season: Double,
			mnth: Double,
			hr: Double,
			holiday: Double,
			weekday: Double,
			workingday: Double,
			weathersit: Double,
			temp: Double,
			atemp: Double,
			hum: Double,
			windspeed: Double,
			cnt: Double)

  def main(args: Array[String]): Unit = {
    SetLogger()
    val spark = SparkSession.builder().appName("Spark SQL basic example").master("local[4]").config("spark.ui.showConsoleProgress","false").getOrCreate()
    import spark.implicits._
    val sch = org.apache.spark.sql.Encoders.product[Data].schema
    
    println("read data")
    val ds = spark.read.format("csv").option("header", "true").schema(sch).load("file:/home/u0756527/Downloads/dataset/hour.csv").as[Data]
    val xy = ds.randomSplit(Array(0.7,0.3))
    val x = xy(0)
    val y = xy(1)
    val dc = ds.drop("instant").drop("dteday").drop("yr").drop("casual").drop("registered").drop("cnt").columns
    val row_ds = ds.select("cnt",dc:_*).as[Data2]
   
    println("setup pipeline")
    val assemblerInputs = row_ds.drop("cnt").columns
    val assembler = new VectorAssembler().setInputCols(assemblerInputs).setOutputCol("t_features")
    val VectorIndexer = new VectorIndexer().setInputCol("t_features").setOutputCol("features")
    val df = new DecisionTreeRegressor().setLabelCol("cnt").setFeaturesCol("features").setImpurity("variance").setMaxDepth(10).setMaxBins(100)
    val pipeline = new Pipeline().setStages(Array(assembler, VectorIndexer, df))
    
    println("train model")
    val pipelineModel = pipeline.fit(x)

    print("predict")
    print("\n")
    val predicted = pipelineModel.transform(y)
    val selected_col = predicted.select("season","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","cnt","prediction")
    selected_col.show(10)

    println("eval model")
    val evaluator = new RegressionEvaluator().setPredictionCol("prediction").setLabelCol("cnt").setMetricName("rmse")
    val predictions =pipelineModel.transform(y)
    val auc= evaluator.evaluate(predictions)
    println(auc)
  }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}