package com.lc.study.no4features.n2transformers

import org.apache.spark.ml.feature.SQLTransformer

/**
 * 顾名思义，可以写sql语句进行一些计算。目前支持如下例子：
 *
    *SELECT a, a + b AS a_b FROM __THIS__
    *SELECT a, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5
    *SELECT a, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b
 *
 */
object SqlTransformerDemo extends App{



  import com.lc.study.spark

  //新建数据集。
  val df = spark.createDataFrame(
    Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")

  //对于对数据集的sql转换。
  val sqlTrans = new SQLTransformer().setStatement(
    "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")
 //查看结果。
  sqlTrans.transform(df).show()



}
