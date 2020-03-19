package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
 * PAC 是一种常用的降维算法，将维度很高的特征向量，变成2-3这个样子的特征向量。
 * 它的主要作用：
 *
 * 缓解维度灾难：PCA 算法通过舍去一部分信息之后能使得样本的采样密度增大（因为维数降低了），这是缓解维度灾难的重要手段；
 * 降噪：当数据受到噪声影响时，最小特征值对应的特征向量往往与噪声有关，将它们舍弃能在一定程度上起到降噪的效果；
 * 过拟合：PCA 保留了主要信息，但这个主要信息只是针对训练集的，而且这个主要信息未必是重要信息。有可能舍弃了一些看似无用的信息，但是这些看似无用的信息恰好是重要信息，只是在训练集上没有很大的表现，所以 PCA 也可能加剧了过拟合；
 * 特征独立：PCA 不仅将数据压缩到低维，它也使得降维之后的数据各特征相互独立；
 *
 * 但是其实现需要很深的数学功底，所以这里先不讲解。
 * 我们来看一下具体的例子。
 * 如果想深入研究：
 * @ https://blog.csdn.net/zouxiaolv/article/details/100590725
 * @ https://zhuanlan.zhihu.com/p/47858230
 */
object PCADemo extends App {

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("PAC").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._


  val data = Array(
    Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
  )
  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

  val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(3)  //设定结果维度。
    .fit(df) //PCA是一个估计器。

  val result = pca.transform(df).select("pcaFeatures")
  result.show(false)





}
