package com.lc.study.no3pipelines


import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}

/**
 * 该demo主要用来进行PipeLine 进行机器学习处理。
 * 包括对pipe模型的持久化。
 */
object PipeDemo extends App{

  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._


  // 准备数据集，包含id、文本数据，标签
  val training = spark.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0)
  )).toDF("id", "text", "label")

  // 分别配置分析器、特征提取、以及逻辑回归。
  val tokenizer = new Tokenizer()
    .setInputCol("text") //这里注意，它真的数据集中的text列
    .setOutputCol("words") //输出列为words
  val hashingTF = new HashingTF()
    .setNumFeatures(1000)
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")  //将words变成向量列。
  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.001)
  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, lr)) //定义stage，也就是定义执行顺序。

  // 对数据集进行训练。
  val model = pipeline.fit(training)

  // 对模型进行持久化
  model.write.overwrite().save("/tmp/spark-logistic-regression-model")

  // 也可以将pipeline持久化。
  pipeline.write.overwrite().save("/tmp/unfit-lr-model")

  //可以对存储起来的模型进行加载
  val sameModel = PipelineModel.load("/tmp/spark-logistic-regression-model")

  //准备测试数据。
  val test = spark.createDataFrame(Seq(
    (4L, "spark i j k"),
    (5L, "l m n"),
    (6L, "spark hadoop spark"),
    (7L, "apache hadoop")
  )).toDF("id", "text")

  //使用模型对测试数据进行预测。
  model.transform(test)
    .select("id", "text", "probability", "prediction")
    .collect()
    .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
      println(s"($id, $text) --> prob=$prob, prediction=$prediction")
    }

}
