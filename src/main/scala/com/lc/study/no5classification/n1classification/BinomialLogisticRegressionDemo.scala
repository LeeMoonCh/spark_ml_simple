package com.lc.study.no5classification.n1classification

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Row, SparkSession}

/*
二项式逻辑回归。
逻辑回归主要用到的是统计学中的一些概念，从判断某件时间产生的概率有多大进行分类。
注：本系列教程不讲解任何公式的推导
大家可以看下面这篇博文来理解逻辑回归
https://blog.csdn.net/garfielder007/article/details/51004991

对于一件事情的发生情况我们有如下认为：
P(yi=1|xi) 其中xi是我们的数据样本。一般我们会设置阀值，当阀值大于0.5时，y=1,当p<0.5时，y=0

而对于非线性分类，我们要做的事情其实很简单，就是将非线性转换为线性。
而我们一般使用的转换函数为：y= 1 / ( 1 + e^(-x) )   ==>这个函数我们称它为 logit函数

更加多的推导看下面的博文:
https://www.sohu.com/a/236530043_466874

 */
object BinomialLogisticRegressionDemo extends App {

  //官方例子 太过于复杂化，我们下面使用一个例子来看一下该逻辑回归的使用方式。
  //我们来看一下个人收入的数据，其中一共分为两类。一类为收入>50K ==> 1 一类为<50K ==> 0
  //首先我们需要把数据格式化成libsvm
  //先看一下libsvm的df格式是什么样子
  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  spark.read.format("libsvm")
    .load("data/simple_libsvm_data")
    .show()
/*
label|            features|
+-----+--------------------+
|  0.0|(692,[127,128,129...|
 */
//也就是一共两列，一列用来记录标签，一列用来记录数据。
val data = spark.sparkContext.textFile("data/shouru50w.csv")
    .filter(f=>{!f.contains("年龄")})
    .map(f=>{
      val datas = f.split(",")


      var sex = 0  //男性
      if(datas(2).trim().equals("Male")){
        sex = 1
      }
      var lable = 0 //小于50k
      if(datas(6).trim.equals(">50K")){
        lable  = 1
      }

      (lable,Vectors.sparse(6,Seq(
        (0,datas(0).trim.toDouble),
        (1,datas(1).trim.toDouble),
        (2,sex.toDouble),
        (3,datas(3).trim.toDouble),
        (4,datas(4).trim.toDouble),
        (5,datas(5).trim.toDouble)
       )
      ))
    }).toDF("label","features")


  //转换成功后，我们就可以新建一个逻辑回归算法进行模型训练。
  val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)




  val lrModel = lr.fit(data)


  //然后对新数据进行预测
  val pre = lrModel.predict(Vectors.sparse(6,Seq(
    (0,34.0),
    (1,11.0),
    (2,1.0),
    (3,0.0),
    (4,0.0),
    (5,55.0)
  )))

  println(pre)










}
