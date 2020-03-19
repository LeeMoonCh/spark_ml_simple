package com.lc.study.no4features.n1extractors

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.SparkSession

/*
CountVectorizer  将文本向量转换成 词频向量，该结果集可以传递给其他算法。如：LDA
它没有太过于特殊的东西，不像Word2Vec 可以做推荐系统，相似度，查重等。
它就是简简单单的将一个已经分过词的文本 进行了词频统计而已。
它在fit(拟合)过程中，会根据vocabSize 也就是词汇量 进行排序输出。
可选参数minDF  用来过滤 词频小于这个值得词。

 */
object CountVectorizerDemo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  val df = spark.createDataFrame(Seq(
    (0, Array("a", "b", "c")),
    (1, Array("a", "b", "b", "c", "a"))
  )).toDF("id", "words")

  // 对数据集集合训练，拿到模型
  val cvModel: CountVectorizerModel = new CountVectorizer()
    .setInputCol("words")
    .setOutputCol("features")
    .setVocabSize(3) //词汇表的最大含量
    .setMinDF(2)
    .fit(df)

//  //也可以这样进行模型。一般不适用。
//  val cvm = new CountVectorizerModel(Array("a", "b", "c"))
//    .setInputCol("words")
//    .setOutputCol("features")

  //进行特征转换。看看结果。
  cvModel.transform(df).show(false)






}
