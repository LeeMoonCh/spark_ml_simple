package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.SparkSession

/**
 * 特征2值化，根据给定阀值进行0-1处理，大于阀值为1，小于阀值为0
 */
object BinaryDemo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._


  val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
  val dataFrame = spark.createDataFrame(data).toDF("id", "feature")

  val binarizer: Binarizer = new Binarizer()
    .setInputCol("feature")
    .setOutputCol("binarized_feature")
    .setThreshold(0.5)  //设置阀值。。。

  val binarizedDataFrame = binarizer.transform(dataFrame)

  println(s"Binarizer output with Threshold = ${binarizer.getThreshold}")
  binarizedDataFrame.show()


}
