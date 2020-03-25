package com.lc.study.no4features.n3featureselectors

import org.apache.spark.ml.feature.RFormula

/*
该算法，支持一下操作：
  ~  类似有等号
  +  拼接
  -  移除
  ： 可以理解成乘号
  .  所有列除了target

  比如有下列数据：
id | country | hour | clicked
---|---------|------|---------
 7 | "US"    | 18   | 1.0
 8 | "CA"    | 12   | 0.0
 9 | "NZ"    | 15   | 0.0


 clicked ~ country + hour

 代表，clicked的结果依赖于country和hour。
 在该算法中，对于字符串列，会进行stringIndexer计算，转换成double，数值的话原封不动。那么结果如下：
 id | country | hour | clicked | features         | label
---|---------|------|---------|------------------|-------
 7 | "US"    | 18   | 1.0     | [0.0, 0.0, 18.0] | 1.0
 8 | "CA"    | 12   | 0.0     | [0.0, 1.0, 12.0] | 0.0
 9 | "NZ"    | 15   | 0.0     | [1.0, 0.0, 15.0] | 0.0

features  ：为该算法求出来的特征向量，label为标签列(和clicked的值一致。)



一下我们我看例子。



 */
object RFormulaDemo extends App{


  import com.lc.study.spark

  val dataset = spark.createDataFrame(Seq(
    (7, "US", 18, 1.0),
    (8, "CA", 12, 0.0),
    (9, "NZ", 15, 0.0)
  )).toDF("id", "country", "hour", "clicked")

  val formula = new RFormula()
    .setFormula("clicked ~ country + hour")  //设置算法规则
    .setFeaturesCol("features")
    .setLabelCol("label")
    .setHandleInvalid("keep")

  val rfModel = formula.fit(dataset)
  rfModel.transform(dataset).select("features", "label").show(false)

  //新建一个数据集。
  val datase1t = spark.createDataFrame(Seq(
    (1, "US", 16, 1.0),
    (2, "CN", 8, 0.0),
    (3, "US", 18, 0.0)  //注意这里。你会发现，索然label结果也是0，但是它转换出来的特征向量和之前的数据是一样的。所以我们可以在这里对这个特征向量进行标签转换。
  )).toDF("id", "country", "hour", "clicked")


  rfModel.transform(datase1t).show(false)














}
