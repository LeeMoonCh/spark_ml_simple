package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.StringIndexer

/*
Spark的机器学习处理过程中，经常需要把标签数据（一般是字符串）转化成整数索引，
而在计算结束又需要把整数索引还原为标签。这就涉及到几个转换器：
StringIndexer、 IndexToString，OneHotEncoder，以及针对类别特征的索引VectorIndexer。

我们直接看demo就行了，这个算法较直观


 */
object StringIndexerdemo extends App{

  import com.lc.study.spark


  val df = spark.createDataFrame(
    Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
  ).toDF("id", "category")

  val indexer = new StringIndexer()
    .setInputCol("category")
    .setOutputCol("categoryIndex")
    .setHandleInvalid("keep") //不去重。如果给 skip 那么就会将模型去重。
  //需要注意，它是一个估计器，需要fit数据集进行训练，产生一个模型。
  val indexedModel = indexer.fit(df)
  val indexed = indexedModel.transform(df)
  indexed.show()

  val df1 = spark.createDataFrame(
    Seq( (0, "b"),(1, "c"),(2, "a"),(3, "c"))
  ).toDF("id", "category")

  //对新数据大标签。
  val preindex = indexedModel.transform(df1)

  preindex.show()






}
