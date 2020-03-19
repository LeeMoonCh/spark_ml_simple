package com.lc.study.no3pipelines

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
 * 该demo主要演示，在不不适用PipeLine的情况下，如何进行转换器、估计器、参数定义
 * 来完成一个小业务。
 */
object NoPipeLineDemo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  // 准备数据样本。第一列为标签列，第二列为特征向量列
  val training = spark.createDataFrame(Seq(
    (1.0, Vectors.dense(0.0, 1.1, 0.1)),
    (0.0, Vectors.dense(2.0, 1.0, -1.0)),
    (0.0, Vectors.dense(2.0, 1.3, 1.0)),
    (1.0, Vectors.dense(0.0, 1.2, -0.5))
  )).toDF("label", "features")

  //这是一个Estimator，是对逻辑回归算法的封装
  val lr = new LogisticRegression()
  //关于逻辑回归的参数设定，这里不做详细讲解，等待做到后面之后，在讲解。
  //最大迭代次数 和回归系数设定。
  lr.setMaxIter(10)
    .setRegParam(0.01)

  //调用fit方法对数据样本进行训练。fit会产生一个模型，该模型也是一个转换器。
  val model = lr.fit(training)


  //定义测试数据：
  val test = spark.createDataFrame(Seq(
    (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
    (0.0, Vectors.dense(3.0, 2.0, -0.1)),
    (1.0, Vectors.dense(0.0, 2.2, -1.5))
  )).toDF("label", "features")

  //解释一下结果中包含的列明含义：
  // label  数据本身的标签
  // features  数据本身特征向量
  //rawprediction 数据在预测结果时，对于不同类别的置信度。
      //probability  预测结果对应不同类别的概率。
  //最终预测结果。
  model.transform(test)
    .show() //查看一下训练结果

  model.transform(test).repartition(1).write.json("output/1.json") //将结果保存起来







}
