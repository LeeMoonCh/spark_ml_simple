package com.lc.study.no1Statistics

import org.apache.spark.ml.linalg.{Vectors,Vector}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.SparkSession

/**
 * 我们先来说术语解释：
 *
 * 平均数、中位数、众数。
 * 样本均值（即n个样本的算术平均值） ，
 * 样本方差（即n个样本与样本均值之间平均偏离程度的度量），
 * 样本极差（样本中最大值减最小值），
 * 众数，样本的各阶原点矩和中心矩。
 *
 * 以上都属于统计量。
 *
 * 没什么可说的，我们直接看例子。
 * 在Spark中，统计量是一个类进行的，它里面定义了一堆隐式转换。方便我们通过
 * df.select进行选择。
 * Summarizer 就是这个类。
 */

object SummarizerDemo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._
  //导入统计量类的隐式转换方法。
//  它提供了以下方法。:
//  *  - mean: 求评价值.
//  *  - variance: 方差.
//  *  - count: 个数统计
//  *  - numNonzeros: 每个数据集中的非0个数。
//  *  - max: 最大值
//  *  - min: 最小值
//  *  - normL2: l2-范数
//  *  - normL1: l1-范数
  import Summarizer._

  val data = Seq(
    (Vectors.dense(2.0, 3.0, 5.0), 1.0),
    (Vectors.dense(4.0, 6.0, 7.0), 2.0)
  ) //新建数据集。该数据集由两个样本组成。

  val df = data.toDF("features", "weight") //将数据集转换成DF，给两列，一列为特征向量，一列为权重标签。

  val (meanVal, varianceVal) = df.select(metrics("mean", "variance") //求某列的平均值以及方差
    .summary($"features", $"weight").as("summary")) //带权重进行计算
    .select("summary.mean", "summary.variance") //将结果拿到
    .as[(Vector, Vector)].first() //返回第一行数据结果。

  //打印 带权重结果。
  println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")
  // mean = [3.333333333333333,5.0,6.333333333333333]
  // [2,3,5],1
  // [4,6,7],2
  //第一列带权重平均值为:(2*1+4*2)/3 = 3.33

  //如果不带权重，可以直接调用统计量中的mean，variance方法进行某一列的统计量计算。
  val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features"))
    .as[(Vector, Vector)].first()

  //不带权重结果。
  println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")

}
