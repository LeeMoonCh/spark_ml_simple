package com.lc.study.no4features.n2transformers


import com.lc.study.spark
import org.apache.spark.ml.feature.DCT
import org.apache.spark.ml.linalg.Vectors


/*
离散余弦变换，一张将时序N的特征向量变成频域N的特征向量的算法。
说直白点就是，将高频特征和低频特征进行区分。可以完成有损压缩。
比如：jpeg图片，就是一种使用DCT的应用。它将图像数据使用DCT算法求出高频数据
和低频数据，在一定范围内丢弃低频数据。然后使用高频数据进行重新编码。
解码后 就和原来的图像在一定程度上是一样的。但是肯定有区别，因为丢失了部分数据。
该算法主要应用在 数字信号、数字图像处理领域
 */
object DiscreteCosineTransformDemo extends App {

  val data = Seq(
    Vectors.dense(0.0, 1.0, -2.0, 3.0),
    Vectors.dense(-1.0, 2.0, 4.0, -7.0),
    Vectors.dense(14.0, -2.0, -5.0, 1.0))

  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

  val dct = new DCT()
    .setInputCol("features")
    .setOutputCol("featuresDCT")
    .setInverse(false)

  val dctDf = dct.transform(df)
  dctDf.select("featuresDCT").show(false)

}
