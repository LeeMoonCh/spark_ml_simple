package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

/*
该算法主要讲数据集中的多个数值列，转换成一个向量列。可以理解成将多个列合并成一列。

这里稍微看一下，和VectorIndexer的不同：
VectorAssembler将多个数值列按顺序汇总成一个向量列。

VectorIndexer将一个向量列进行特征索引，一般决策树常用。

VectorIndexer对于离散特征的索引是基于0开始的。其不保证对每个值每次索引建立的索引值都一样，但是会保证对于0值总是会给索引值0。


 */
object VectorAssemblerDemo extends App{

  import com.lc.study.spark



  val dataset = spark.createDataFrame(
    Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
  ).toDF("id", "hour", "mobile", "userFeatures", "clicked")

  val assembler = new VectorAssembler()
    .setInputCols(Array("hour", "mobile", "userFeatures"))
    .setOutputCol("features")

  val output = assembler.transform(dataset)
  println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column 'features'")
  output.select("features", "clicked").show(false)



}
