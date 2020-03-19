package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vectors

/*
标准化数据。。。  很直观，没有什么讲的。。和归一化一样。
作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。

并不是所有的标准化都能给estimator带来好处。
 */
object StandardScalarDemo extends App{

  import com.lc.study.spark

  val dataFrame = spark.read.format("libsvm").load("data/simple_libsvm_data")

  val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)  //使用方差。
    .setWithMean(false) //不适用均值。

//  // Compute summary statistics by fitting the StandardScaler.
//  val scalerModel = scaler.fit(dataFrame)
//
//  // Normalize each feature to have unit standard deviation.
//  val scaledData = scalerModel.transform(dataFrame)
//  scaledData.show()

  //官方数据集过大，看着不太明显，我们使用一些例子来看。
  val data1 = spark.createDataFrame(Seq(
    (1,Vectors.sparse(3,Array(0,1,2),Array(2.0,5.0,7.0))),
    (0,Vectors.sparse(3,Array(0,1,2),Array(3.0,5.0,9.0))),
    (0,Vectors.sparse(3,Array(0,1,2),Array(4.0,7.0,9.0))),
    (1,Vectors.sparse(3,Array(0,1,2),Array(2.0,4.0,9.0))),
    (1,Vectors.sparse(3,Array(0,1,2),Array(9.0,5.0,7.0))),
    (0,Vectors.sparse(3,Array(0,1,2),Array(2.0,5.0,9.0))),
    (2,Vectors.sparse(3,Array(0,1,2),Array(3.0,4.0,9.0))),
    (1,Vectors.sparse(3,Array(0,1,2),Array(8.0,4.0,9.0))),
    (0,Vectors.sparse(3,Array(0,1,2),Array(3.0,6.0,2.0))),
    (1,Vectors.sparse(3,Array(0,1,2),Array(5.0,9.0,2.0)))
  )).toDF("label","features")

  scaler.fit(data1).transform(data1).show(false)


}
