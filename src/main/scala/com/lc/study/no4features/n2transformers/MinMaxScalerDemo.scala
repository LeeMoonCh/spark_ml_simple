package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.{MaxAbsScaler, MinMaxScaler}
import org.apache.spark.ml.linalg.Vectors

/*
也是一种归一化算法，这种归一化是将特征向量归一到[0,1]之间。
归一化 在很多算法中很有必要的。具体的可以看如下博文：
https://blog.csdn.net/weixin_40683253/article/details/81508321?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
 在spark中，是可以设置归一区间的，也就是[min,max] 默认是min=0,max=1


 */
object MinMaxScalerDemo extends App{

  import com.lc.study.spark


  //设置数据。
  val dataFrame = spark.createDataFrame(Seq(
    (0, Vectors.dense(1.0, 0.1, -1.0)),
    (1, Vectors.dense(2.0, 1.1, 1.0)),
    (2, Vectors.dense(3.0, 10.1, 3.0))
  )).toDF("id", "features")

  val scaler = new MinMaxScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")

  // 该算法是一个估计器，会产生一个模型。
  val scalerModel = scaler.fit(dataFrame)

  //所有特征值将会被归一在[0,1]区间内。
  val scaledData = scalerModel.transform(dataFrame)
  println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
  scaledData.select("features", "scaledFeatures").show()

  //还有一个归一化算法，就是默认将所有数据，归一到[-1,1]之间。
  val maxAbsScaler = new MaxAbsScaler()
    .setInputCol("features")
    .setOutputCol("maxAbsFeatures")

  maxAbsScaler.fit(dataFrame).transform(dataFrame).show()





}
