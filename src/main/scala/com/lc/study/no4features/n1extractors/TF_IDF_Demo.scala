package com.lc.study.no4features.n1extractors

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession


/*
@ 以下部分内容引自百度百科：
TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。

IDF的主要思想是：如果包含词条t的文档越少，也就是n越小，IDF越大，则说明词条t具有很好的类别区分能力。如果某一类文档C中包含词条t的文档数为m，
而其它类包含t的文档总数为k，显然所有包含t的文档数n=m+k，当m大的时候，n也大，按照IDF公式得到的IDF的值会小，就说明该词条t类别区分能力不强。
但是实际上，如果一个词条在一个类的文档中频繁出现，则说明该词条能够很好代表这个类的文本的特征，这样的词条应该给它们赋予较高的权重，并选来作为
该类文本的特征词以区别与其它类文档。
故TF（词频）就和IDF一起使用，用来评估一个单词对于文件的重要程度。

TF-IDF倾向于过滤掉常见的词语，保留重要的词语。(画黑板，这个算法的主要作用。)

啥也先不说，先上以下在官网中对于IDF以及TF的定义：
先说一些概念：词 term 用t进行表示。 文档 document 用d表示，文集/语料库 使用 D进行表示
那么 TF(t,d) 代表的意思为：特定单词t在某一个文档d中出现的次数。所以TF就是词频的意思。
那么 DF(t,D) 代表的意思为：特定单词t在某一个文集D中出现过的文档个数。比如：那么 DF(t,D) = 100

如果使用TF来进行单词t对于文档的重要性特征提取，那么一些常用词：比如this、of等无意义单词的权重肯定很高。对于文档分类或者文档主题确定而言
没有任何意义。所以，使用IDF进行这些单词的过滤。
看公式：
  IDF(t,D)=log [ (|D|+1) / (DF(t,D)+1) ]
  |D| => 文集中文档的个数。
  举个例子：
  比如：现在|D| = 10000
  t = this
  this 应该是一个常用词，在大多数文章中应该都会出现。故：
  DF(t,D) = 10000
  IDF(t,D) = log(10001/10001) = log(1) = 0

  TF-IDF = TF(t,d)*IDF(t,D)
  那么this通过TF-IDF算出来的权重为：0

  以上只是一种TF-IDF的实现，在机器学习领域，TF和IDF都有不同的实现。spark中对于TF-IDF并不是抽象出一个转换器，而是
  TF和IDF分开。它们可以进行灵活组合。
  TF：Spark提过了HashingTF 和 CountVectorizer 来对文档进行词频向量的生成。
  其中HashingTF 使用了特征哈希（https://en.wikipedia.org/wiki/Feature_hashing）并使用MurmurHash-3作为hash
  方法。索引向量维度也就是hash桶的个数，最好是2的n次方。

  CountVectorizer  就相对来说比较简单，直接将文档转换成词统计向量。
  TF是一个转换器。

  IDF：它是一个估计器、也就是说它用来训练模型。它可以生产模型，并对TF转换出来的DF，进行进一步转换。（因为模型是一个转换器）。
  最终，TF-IDF就完成了。简单而言，TF-IDF是将常用词进行降权处理，并将可以进行文档问了的单词进行了加权处理。
  然后，我们看例子。
 */
object TF_IDF_Demo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._
  val sentenceData = spark.createDataFrame(Seq(
    (0.0, "Hi I heard about Spark and i like spark"),
    (0.0, "I wish Java could use case classes like"),
    (1.0, "Logistic regression models are neat")
  )).toDF("label", "sentence") //嗯。。。这个标签目前我不知道有什么用。

  //使用分词转换器，将DF转换成一个个单词。
  val tk = new Tokenizer().setInputCol("sentence").setOutputCol("words")

  //对raw 数据进行转换。
  val wdata = tk.transform(sentenceData)

  wdata.show() //我们看看转换结果。
  /*
  +-----+--------------------+--------------------+
|label|            sentence|               words|
+-----+--------------------+--------------------+
|  0.0|Hi I heard about ...|[hi, i, heard, ab...|
|  0.0|I wish Java could...|[i, wish, java, c...|
|  1.0|Logistic regressi...|[logistic, regres...|
+-----+--------------------+--------------------+
   */

  //我们使用HashingTF 进行词频统计
  val htf = new HashingTF().setInputCol("words").setOutputCol("wordf")
    .setNumFeatures(16)  //设定特征维度，默认为2^18 因为我们是demo，所以这里可以适当减少，比如2^4

  //对单词数据集进行hashingTF的转换
  val wordfData = htf.transform(wdata)

  wordfData.show() //看看结果集~
  /*
  +-----+--------------------+--------------------+--------------------+
|label|            sentence|               words|               wordf|
+-----+--------------------+--------------------+--------------------+
|  0.0|Hi I heard about ...|[hi, i, heard, ab...|(16,[1,8,13],[3.0...|
|  0.0|I wish Java could...|[i, wish, java, c...|(16,[1,5,6,9,15],...|
|  1.0|Logistic regressi...|[logistic, regres...|(16,[2,4,7,9,14],...|
+-----+--------------------+--------------------+--------------------+
   */

  //新建一个IDF 用来进行模型训练
  val idf = new IDF().setInputCol("wordf").setOutputCol("features")

  //训练
  val idfMode = idf.fit(wordfData)

  //对数据进行转换，拿到最后的特征向量。
//  idfMode.transform(wordfData).write.json("output/tf_idf") //将结果进行保存。
//    idfMode.transform(wordfData)
//    .select("words","features").collect().foreach(println)
  /*
  [WrappedArray(hi, i, heard, about, spark),(16,[1,8,13],[0.8630462173553426,0.6931471805599453,0.6931471805599453])]
[WrappedArray(i, wish, java, could, use, case, classes),(16,[1,5,6,9,15],[0.5753641449035617,0.6931471805599453,0.6931471805599453,0.28768207245178085,1.3862943611198906])]
[WrappedArray(logistic, regression, models, are, neat),(16,[2,4,7,9,14],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.28768207245178085,0.6931471805599453])]
   */

  //我们看结果，发生了hash碰撞，比如在第一个文档中，应该有5个单词分布在16个桶的5个里，但是，这5个单词 却分布在了3个桶里。。。
  //所以我们加大桶数，
  htf.setNumFeatures(2000)

  val wordfData1 = htf.transform(wdata)

  idf.fit(wordfData1).transform(wordfData1)
    .select("words","features","wordf").collect().foreach(println)

  /*
  [WrappedArray(hi, i, heard, about, spark),(2000,[1105,1329,1357,1777,1960],[0.6931471805599453,0.28768207245178085,0.6931471805599453,0.6931471805599453,0.6931471805599453])]
[WrappedArray(i, wish, java, could, use, case, classes),(2000,[213,342,489,495,1329,1809,1967],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.28768207245178085,0.6931471805599453,0.6931471805599453])]
[WrappedArray(logistic, regression, models, are, neat),(2000,[286,695,1138,1193,1604],[0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453,0.6931471805599453])]
   */

  //我们单独看第一行结果，发现 hi i 中，i的结果就很低，所以i就可以被我们抛弃掉~不作为文章分类的要素。
  //在实际应用过程中，TF-IDF 是将词频高的词，拿出来，然后做IDF扩展，将词频高的词而TF-IDF低的词进行过滤，保留
  //词频高、TF-IDF高的词，作为文章的分类的重要依据。



















}
