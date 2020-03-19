package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors

/*
特征归一化，对特征向量进行p-norm计算，也就是范数计算，得到一个l-n范数，默认为l2-范数。p的值可以进行指定。
 */
object NormalizerDemo extends App{

  import com.lc.study.spark

  //数据。
  val dataFrame = spark.createDataFrame(Seq(
    (0, Vectors.dense(1.0, 0.5, -1.0)),
    (1, Vectors.dense(2.0, 1.0, 1.0)),
    (2, Vectors.dense(4.0, 10.0, 2.0))
  )).toDF("id", "features")

  // 求1 范数。
  val normalizer = new Normalizer()
    .setInputCol("features")
    .setOutputCol("normFeatures")
    .setP(1.0)

  val l1NormData = normalizer.transform(dataFrame)
  println("Normalized using L^1 norm")
  l1NormData.show()

  // 将p改成 无限大。。。。
  val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
  println("Normalized using L^inf norm")
  lInfNormData.show()








}
