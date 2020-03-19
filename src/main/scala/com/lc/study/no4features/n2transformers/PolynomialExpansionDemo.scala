package com.lc.study.no4features.n2transformers

import com.lc.study
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors

/*
多项式扩展。
多项式：多个单项式的和组成的代数式 为一个多项式 如：1a+2b+3c+1
单项式：为整式，有字母和数字的✖️的表达式。如：1 ，a ,-2c等等。

spark的多项式扩展，是将你的特征向量进行多现实转换的算法。可以理解成一种特征扩展或者升维手段。
@引用 https://segmentfault.com/a/1190000014920546?utm_source=channel-newest
该文章说的到位，`如果随时间每一阶变化率（每一阶导数）都一样，那这俩曲线肯定是完全等价的`
也就是说 通多多项式扩展的出的特征，在线性模型上看，可以认为是一样的。（这就是意义）
参数一般设置：degree 多项式次数，也就是 扩展多少次。

 */
object PolynomialExpansionDemo extends App {

  val spark = study.spark

  spark.sparkContext.setLogLevel("warn")

  import spark.implicits._

  //定义数据，当前数据为2维特征向量
  val data = Array(
    Vectors.dense(2.0, 1.0),
    Vectors.dense(0.0, 0.0),
    Vectors.dense(3.0, -1.0)
  )
  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

  val polyExpansion = new PolynomialExpansion()
    .setInputCol("features")
    .setOutputCol("polyFeatures")
    .setDegree(3)  //多项式3介。   该算法旨在解决特征欠拟合问题。


  val polyDF = polyExpansion.transform(df)
  polyDF.show(false)
















}
