package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.ml.linalg.Vectors

/*
对特征向量 进行乘积操作，啥意思，比如你想对某一个特征向量进行同维度的向量相乘，就可以使用该算法。
这在根据不同权重重新求一个特征向量很有用。
 */
object ElementwiseProductDemo extends App{

  import com.lc.study._
  val dataFrame = spark.createDataFrame(Seq(
    ("a", Vectors.dense(1.0, 2.0, 3.0)),
    ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")

  //需要乘积的另外一个同维度的特征向量。
  val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
  val transformer = new ElementwiseProduct()
    .setScalingVec(transformingVector)
    .setInputCol("vector")
    .setOutputCol("transformedVector")

  // 每行都会和transformingVector 进行矩阵相乘操作。
  transformer.transform(dataFrame).show()


}
