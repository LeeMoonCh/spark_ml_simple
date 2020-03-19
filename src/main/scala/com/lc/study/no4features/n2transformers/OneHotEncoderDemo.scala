package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer}
/*
one-hot编码器，主要作用用来对多个维度的标签向量进行自动分类。也就是数字化。
比如：
有如下三个特征属性：

    性别：["male"，"female"]
    地区：["Europe"，"US"，"Asia"]
    浏览器：["Firefox"，"Chrome"，"Safari"，"Internet Explorer"]

我们的分类想根据 性别，地区，浏览器三个维度进行分类，如果简单的按照每个维度的小标进行标记，对于
["male"，"US"，"Internet Explorer"] 这样的数据，结果应该为：
在spark中，对于string类型的标签数据，应该先试用stringindex进行数字转换。

 */
object OneHotEncoderDemo extends App {

  import com.lc.study.spark

//  val df = spark.createDataFrame(Seq(
//    (0.0, 1.0),
//    (1.0, 0.0),
//    (2.0, 1.0),
//    (0.0, 2.0),
//    (0.0, 1.0),
//    (2.0, 0.0)
//  )).toDF("categoryIndex1", "categoryIndex2")
//
//  val encoder = new OneHotEncoderEstimator()
//    .setInputCols(Array("categoryIndex1", "categoryIndex2"))
//    .setOutputCols(Array("categoryVec1", "categoryVec2"))
//  val model = encoder.fit(df)
//
//  val encoded = model.transform(df)
//  encoded.show()

  //以上代码看着不太直观，我们使用如下这个例子：
  //[log,text,soyo,hadoop]
  //[a,b,c]
  val df=spark.createDataFrame(Seq(
    (0,"log","a"),
    (1,"text","b"),
    (2,"text","b"),
    (3,"soyo","c"),
    (4,"text","c"),
    (5,"log","a"),
    (6,"log","b"),
    (7,"log","c"),
    (8,"hadoop","a")
  )).toDF("id","label1","label2")


  val n1Model = new StringIndexer().setInputCol("label1").setOutputCol("n1").fit(df)
  val n2Model = new StringIndexer().setInputCol("label2").setOutputCol("n2").fit(df)

  val dfn12 = n2Model.transform(n1Model.transform(df))


  val nHeModel = new OneHotEncoderEstimator()
    .setInputCols(Array("n1","n2"))
    .setOutputCols(Array("v1","v2")).fit(dfn12)

  nHeModel.transform(dfn12).show(false) //先看一下本身数据的转换结果。

  //新建数据集。用之前的nHeModel进行标签数字化。
  val df1=spark.createDataFrame(Seq(
    (0,"log","a"),
    (1,"text","b"),
    (2,"text","a"),
    (4,"hadoop","b") //这个在之前的模型中不存在。
  )).toDF("id","label1","label2")

  val df1n12 = n2Model.transform(n1Model.transform(df1))

  nHeModel.transform(df1n12).show(false) //再看一下新数据的转换结果。对比一下不同。
  //你会发现新数据中的最后一个元素的转换结果不一样了。







}
