package com.lc.study.no4features.n1extractors

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession}

/*
 @部分内容引用自百度百科：
 Word2vec 可以根据给定的语料库，通过优化后的训练模型快速有效地将一个词语表达成向量形式，为自然语言处理领域的应用研究提供了新的工具。
 @以下内容引用知乎：
 https://x-algo.cn/index.php/2016/03/12/281/
 上面的这篇文章写的很好。
 具体原理实现，恕我根本就看不懂~~~~

好了，我们来看一下官方给出的Demo
 */
object Word2VecDemo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  // 将输入文本，转换成单词序列。
  val documentDF = spark.createDataFrame(Seq(
    "Hi I heard about Spark".split(" "),
    "I wish Java could use case classes".split(" "),
    "Logistic regression models are neat".split(" ")
  ).map(Tuple1.apply)).toDF("text")

  documentDF.foreach(f=>{
    println(f.toString())
  }) //看一下结果集~

  // 定义Word2Vec
  val word2Vec = new Word2Vec()
    .setInputCol("text")
    .setOutputCol("result")
    .setVectorSize(5)
    .setMinCount(0)
  //基于文档数据进行训练。获得模型。
  val model = word2Vec.fit(documentDF)

  //使用模型进行特征转换。将单词转换成词向量。
  val result = model.transform(documentDF)

  //使用模型，进行和I相关的词查询。~(使用这个算法的确可以进行推荐系统的设定。~)
  model.findSynonyms("I",2).show()


  println("----")
  result.foreach(f=>println(f))

//  result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
//    println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }

  //@  https://blog.csdn.net/hjj974834257/article/details/79089686  有应用场景的具体实现。



}
