package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.QuantileDiscretizer

/*
分位数离散化，其实也是一种针对连续数据的离散化，将连续数据(不具备分类特征的)列，通过分位数进行求值。
值一样的数据属于同一个桶里，可以理解成hash。
所以，该算法需要我们设定桶数，官网中的例子非常直接明了，故不多解释。(也不难理解)
 */
object QuantileDiscretizerDemo extends App{

  import com.lc.study.spark

  //新建数据集
  val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
  val df = spark.createDataFrame(data).toDF("id", "hour")

  //定义分位数离散
  val discretizer = new QuantileDiscretizer()
    .setInputCol("hour")  //对hour列进行离散
    .setOutputCol("result")
    .setNumBuckets(3) //分为3个桶。也就是分为3类

  //该算法是一个估计器。模型可用于其他数据。
  val result = discretizer.fit(df).transform(df)
  result.show(false)




}
