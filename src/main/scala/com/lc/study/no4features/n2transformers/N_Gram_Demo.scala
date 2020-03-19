package com.lc.study.no4features.n2transformers

import com.huaban.analysis.jieba.JiebaSegmenter
import com.lc.study.Utils
import com.lc.study.no4features.n2transformers.TokenizerDemo.spark
import org.apache.spark.ml.feature.{CountVectorizer, NGram}
import org.apache.spark.sql.SparkSession

/*
一种对文本特征转换的技巧，具体而言就是 将文本按照字节进行处理，到文本中就是根据词就行处理。然后根据N，N为我们自己设定的为窗口长度进行滑动。没滑动一次
生成一个Gram。在进行一些过滤技巧，生成了有前后文关系的N-Gram关系链。
该链条有什么作用，比如可以判断某一句话是否合理。在搜索引擎中，我们打一个词后，它会给我们很多相关选项供我们使用。也是
N-Gram的一种使用。

@ https://zhuanlan.zhihu.com/p/32829048  该篇文章 对于N-Gram的讲解非常到位，并且有例子可看。

一般 n在1~3之间
进行N_gream训练之前，需要对数据进行分词操作。最好不要过滤停用词

可以去搜狗实验室，把新闻数据下载下来，然后进行训练，我们随便弄点新的句子看一下结果如何。
 */
object N_Gram_Demo extends App{



  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  val wordDataFrame = spark.createDataFrame(Seq(
    (0, Array("Hi", "I", "heard", "about", "Spark")),
    (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
    (2, Array("Logistic", "regression", "models", "are", "neat"))
  )).toDF("id", "words")

  val ngram = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")

  val ngramDataFrame = ngram.transform(wordDataFrame)

  ngramDataFrame.show(false)


  val s = spark.sparkContext.textFile("/Users/lc/Downloads/news_tensite_xml.smarty.dat")
    .filter(f=>f.contains("<content>"))
    .map(f=>f.replaceAll("<content>","").replaceAll("</content>",""))
    .filter(f=>f!="")
    .zipWithIndex()
    .map(f=>(f._2.toInt,f._1))
    .toDF("id","s")



//  s.printSchema()


  //使用自定义分词器进行分词


//  val sentenceDataFrame1 = spark.createDataFrame(Seq(
//    (0, "这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。"),
//    (1, "我不喜欢日本和服。"),
//    (2, "雷猴回归人间。"),
//    (3,"工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作。"),
//    (4,"结果婚的和尚未结过婚的")
//  )).toDF("id", "s")

//  sentenceDataFrame1.printSchema()

  val znTokenizer = new ZnTokenizer().setInputCol("s").setOutputCol("words")

  val words = znTokenizer.transform(s)


//
//  words.show(false)

  //使用n-gram进行bigram
  val NGram = new NGram().setInputCol("words").setOutputCol("ngram").setN(2)

  val n_gram_words = NGram.transform(words)


  //看看结果。
//  n_gram_words.show(false)
  //可以将语料库存起来。







}
