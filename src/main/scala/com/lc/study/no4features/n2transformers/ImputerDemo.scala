package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.Imputer

/*
该估计器 是用来对数据集中的缺失值进行默认补全。使用所在列的中位数或者平均值 进行补全。
官网的例子也很直白明了，故不多做解释，直接运行代码，看结果即可。
 */
object ImputerDemo extends App{

  import com.lc.study.spark

  //新建数据集，数据集中有空值。
  val df = spark.createDataFrame(Seq(
    (1.0, Double.NaN),
    (2.0, Double.NaN),
    (Double.NaN, 3.0),
    (4.0, 4.0),
    (5.0, 5.0)
  )).toDF("a", "b")

  val imputer = new Imputer()
    .setInputCols(Array("a", "b")) //对 a,b进行补全。
    .setOutputCols(Array("out_a", "out_b"))

  val model = imputer.fit(df)
  model.transform(df).show()



}
