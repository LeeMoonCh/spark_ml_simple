package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.{VectorAssembler, VectorSizeHint}
import org.apache.spark.ml.linalg.Vectors

/*
向量大小确定 算法。
很多时候，对于向量属性的列进行大小确定是很有用的。比如，在VectorAssembler算法中，输出列的
大小和元信息就是根据其输入列的大小进行确定的。

该算法允许我们 在数据流没有开始前就对某一向量的大小进行牟定，以方便后续算法直接使用。
我们必须指定一个输入列和大小来确定该算法。
具体作用可以说是用来过滤数据。
我们看一下例子。
 */
object VectorSizeHintDemo extends App{

  import com.lc.study.spark


  //定义数据集
  val dataset = spark.createDataFrame(
    Seq(
      (0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0),
      (0, 18, 1.0, Vectors.dense(0.0, 10.0), 0.0))
  ).toDF("id", "hour", "mobile", "userFeatures", "clicked")

  dataset.show() //看一下结果。

  val sizeHint = new VectorSizeHint()
    .setInputCol("userFeatures")
    .setHandleInvalid("skip")  //对数据集中的数据中，包含缺失值和空值的数据 进行丢弃操作。
    .setSize(3)

  val datasetWithSize = sizeHint.transform(dataset)
  println("Rows where 'userFeatures' is not the right size are filtered out")
  datasetWithSize.show(false)

  //新建一个多列转1列算法。
  val assembler = new VectorAssembler()
    .setInputCols(Array("hour", "mobile", "userFeatures"))
    .setOutputCol("features")

  // 对原数据进行转换。
  val output = assembler.transform(datasetWithSize)
  //看看结果。发现 经过第一步过滤，将向量维度为2的那条数据过滤了，保证算法可被正常执行。
  println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
  output.select("features", "clicked").show(false)


}
