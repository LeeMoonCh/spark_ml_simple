package com.lc.study.no4features.n1extractors

import org.apache.spark.ml.feature.FeatureHasher
import org.apache.spark.sql.SparkSession

/*
特征哈希，在TF-IDF中其实已经接触过了。对就是hashingTF。
对他的讲解非常通俗易懂的情况如下博文：
@ https://www.jianshu.com/p/9c40b8dc60bf

应用场景之一，垃圾分类。比如垃圾邮件，我们可以使用特征哈希对上万甚至上10万篇文本进行
哈希特征处理，然后打入标签。这些标签 只有0和1 也就是2分类应用中的场景。
训练出来模型后，可以对着10万篇邮件进行 只用特征向量和标签，然后使用相同的模型对某一篇
新的邮件进行特征哈希。使用算法完成预测~~~不管你使用贝叶斯还是逻辑回归都可以。~
一下，我们来看一下官方给的例子。
 */
object FeatureHasherDemo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  val dataset = spark.createDataFrame(Seq(
    (2.2, true, "1", "foo"),
    (3.3, false, "2", "bar"),
    (4.4, false, "3", "baz"),
    (5.5, false, "4", "foo")
  )).toDF("real", "bool", "stringNum", "string")

  val hasher = new FeatureHasher()
    .setInputCols("real", "bool", "stringNum", "string")
    .setOutputCol("features")

  val featurized = hasher.transform(dataset)
  featurized.show(false)




}
