package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession

/*
停用词过滤，对于英文而言类似于：is a of
对于中文而言，就是我的，你的，是，的等等。很多时候我们并不需要这样的数据。
所以，完全可以使用停用词过滤 将这些词进行过滤。
spark提过的是英文的，想用中文的，可以直接在中文分词中完成。此处不对停用词讲解。较简单。
 */
object StopWordsRemoverDemo extends App{
  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  val remover = new StopWordsRemover()
    .setInputCol("raw")
    .setOutputCol("filtered")

  val dataSet = spark.createDataFrame(Seq(
    (0, Seq("I", "saw", "the", "red", "balloon")),
    (1, Seq("Mary", "had", "a", "little", "lamb"))
  )).toDF("id", "raw")

  remover.transform(dataSet).show(false)

}
