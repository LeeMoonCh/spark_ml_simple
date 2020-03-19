package com.lc.study.no1Statistics

import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{Row, SparkSession}

/**
 * Correlation  统计学中的相关性统计 在spark.ml中的使用。
 * 在spark中，相关性统计是通过  Correlation 类来完成的。
 * 现在，开始说一些专业术语：@一下内容引自百度百科
 *
 * 相关系数：考察两个事物（在数据里我们称之为变量）之间的相关程度。
 * 如果有两个变量：X、Y，最终计算出的相关系数的含义可以有如下理解：
 * (1)、当相关系数为0时，X和Y两变量无关系。
 * (2)、当X的值增大（减小），Y值增大（减小），两个变量为正相关，相关系数在0.00与1.00之间。
 * (3)、当X的值增大（减小），Y值减小（增大），两个变量为负相关，相关系数在-1.00与0.00之间。
 *
 * 通常情况下通过以下取值范围判断变量的相关强度：
 * 　　相关系数 0.8-1.0 极强相关
 * 　　0.6-0.8 强相关
 * 　　0.4-0.6 中等程度相关
 * 　　0.2-0.4 弱相关
 * 　　0.0-0.2 极弱相关或无相关
 *
 * OK,下面我们来说一下机器学习中的一些专业术语：
 *  向量：行向量 你可以认为成只有一行的数据集。类似于一维数组。(但是需要注意，如果从矩阵来看，向量应该是一个列矩阵)
 *  比如：x = (1,2)  就可以认为是一个二维向量，x=(1,2,3)就是一个三维向量。以此类推。
 *  矩阵：多个行向量组成，你可以认为是多维数组。
 *
 * OK，下面我们通过一些数据集来计算一下 两个数据集之间的相关性。
 */
object CorrelationDemo extends App{


  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  //需要注意的是，在机器学习中，数据集一般我们都使用矩阵和向量表示。故
  //Matrix, Vectors 我们需要对这两个类进行研究。Matrix是矩阵类，Vectors是向量类。
  //我们先研究向量。
  //sparse方法，字面意思为稀疏向量。方法有两个参数，一个为该向量的长度，一个为该向量的稀疏值。
  //所谓稀疏，就是向量长度大于值。下面这个就是产生了一个 (0,1.0),(2,0),(3,-2.0),(4,0)的稀疏向量。
  val vec1 = Vectors.sparse(4,Seq((0,1.0),(3,-2.0)))
  println(vec1.toString)  //结果为 (4,[0,3],[1.0,-2.0])  这里的意思是，4位长度，[0,3]为index，[1.0,-2.0]为对应的下标值

  //dense 正常向量，会根据参数生成一个参数个数长度的向量。
  val vec2 = Vectors.dense(4.0,5.0, 0.0, 3.0)
  println(vec2.toString)  //[4.0,5.0,0.0,3.0] 和稀疏向量不同，这种向量完全就是一个数组。


  //norm  范数函数，用来求p-范数。
  //@一下内容参考自：知乎：https://zhuanlan.zhihu.com/p/26884695
  //范数可以理解成一种距离的表示。
  //欧式范数 对于线性代数中的向量  x = (3,4) 而言，它的 l2-范数 = sqrt( pow(3,2) + pow(4,2) )
  //spark 这里，是lp-范数。  即对于向量 x = (3,4,5,-1,3)的 p范数为：pow(3,p)+pow(4,p)+pow(-1,p)+pow(3,p)之和
  //的p跟方的值。如果这里你传入2，那么就是l2-范数，也就是欧式距离。
  val l2_fs = Vectors.norm(Vectors.dense(3,4),2) //我们算一下二维向量(3,4)的l2-范数，就是5
  println(l2_fs)  //5

  //定义一个n维 所有值为0的向量。
  Vectors.zeros(4)

  //到此，我们的所有Vectors的方法说完。
  //返回到官网提供的例子。我么看：
  val data = Seq(
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    Vectors.dense(4.0, 5.0, 0.0, 3.0),
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
  )  //定义数据集，该数据集由多个向量组成，其中0，3为稀疏向量，1，2为正常向量。

  val df = data.map(Tuple1.apply).toDF("features") //基于数据集，进行df创建。

  df.show() //我们查看一下结果。结果显示，它只有1列，并且每个值都为向量本身。
//  +--------------------+
//  |            features|
//  +--------------------+
//  |(4,[0,3],[1.0,-2.0])|
//  |   [4.0,5.0,0.0,3.0]|
//  |   [6.0,7.0,0.0,8.0]|
//  | (4,[0,3],[9.0,1.0])|
//  +--------------------+


  //求皮尔森（pearson）相关系数
  //一下引用 https://blog.csdn.net/ruthywei/article/details/82527400
  //上面我们知道，相关系数是两个数据集，在其范围内的正相关于负相关值。
  //Correlation.corr的默认使用皮尔森相关系数进行计算，它是将某一个向量集中的向量 1，1进行皮尔森相关系数求解。
  //因为我们的df 中只有1列，所以该方法对该列进行了4*4 的皮尔森系数求解。
//    +--------------------+
  //  |            features|
  //  +--------------------+
  //  |(4,[0,3],[1.0,-2.0])|
  //  |   [4.0,5.0,0.0,3.0]|
  //  |   [6.0,7.0,0.0,8.0]|
  //  | (4,[0,3],[9.0,1.0])|
  //  +--------------------+
  // (4,[0,3],[1.0,-2.0])  分别和 它自身，[4.0,5.0,0.0,3.0]，[6.0,7.0,0.0,8.0]，(4,[0,3],[9.0,1.0])
  //进行皮尔森系数求解。故结果是四个，就是第一行的结果。
  //之后以此类推。
  //皮尔森系数为线性相关系数。
  val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
  println(s"Pearson correlation matrix:\n $coeff1")

  //这是再求，df集 某一列 的斯皮尔曼相关系数。 斯皮尔曼系数为等级相关系数。
  val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
  println(s"Spearman correlation matrix:\n $coeff2")
  // $example off$

  spark.stop()





}
