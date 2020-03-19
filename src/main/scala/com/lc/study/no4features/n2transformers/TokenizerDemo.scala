package com.lc.study.no4features.n2transformers

import com.huaban.analysis.jieba.JiebaSegmenter
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}
/*
Tokenizer  分词器，很不幸的是，Tokenizer是spark提供的内部分词器。只适合于英文文本。。。。。
 */

/*
自定义中文分词器，使用jieba分词器
 */
class ZnTokenizer(override val uid:String)
  extends UnaryTransformer[String, Seq[String], ZnTokenizer] with DefaultParamsWritable with Serializable {

  import com.huaban.analysis.jieba.JiebaSegmenter
  //java scala之间的集合互转，必须用的。
  import scala.collection.JavaConverters._

  //一套组合拳，解决第三方依赖类的序列化问题。
  @transient
  lazy val segmenter = new JiebaSegmenter
  //空构造。
  def this() = this(Identifiable.randomUID("zntok"))
  override protected def createTransformFunc: String => Seq[String] = { f=>


    val listword = segmenter.sentenceProcess(f)
    listword.asScala
  }

  override protected def outputDataType: DataType =  new ArrayType(StringType, true)
}





object TokenizerDemo extends App{

  //创建程序入口,sparkSession
  val spark = SparkSession.builder().appName("CorrelationDemo").master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")
  //导入 隐式转换，方便将数据集转换成DF
  import spark.implicits._

  val sentenceDataFrame = spark.createDataFrame(Seq(
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
  )).toDF("id", "sentence")

  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
  val regexTokenizer = new RegexTokenizer()
    .setInputCol("sentence")
    .setOutputCol("words")
    .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

  import org.apache.spark.sql.functions._

  //等于这里做了一个简单的udf
  val countTokens = udf { (words: Seq[String]) => words.length }

  val tokenized = tokenizer.transform(sentenceDataFrame)

  //查看一下通过tokenizer 的分词结果。
  tokenized.show(false)

//  tokenized.select("sentence", "words")
//    .withColumn("tokens", countTokens(col("words"))).show(false)

  val regexTokenized = regexTokenizer.transform(sentenceDataFrame)

  //查看一下正则分词器的分词结果。
  regexTokenized.show(false)

//  regexTokenized.select("sentence", "words")
//    .withColumn("tokens", countTokens(col("words"))).show(false)


  //分词器总体来说，没有什么好说的，但是如果我们用的是中文，那么可以根据官方提供的分词器，自己写一个
  //中文分词器的 transfomer。 中文分词器有很多。可以自己加入依赖进行代码编写。

  //下面我们来试一下中文分词器。


  val sentenceDataFrame1 = spark.createDataFrame(Seq(
    (0, "这是一个伸手不见五指的黑夜。我叫孙悟空，我爱北京，我爱Python和C++。"),
    (1, "我不喜欢日本和服。"),
    (2, "雷猴回归人间。"),
    (3,"工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作。"),
    (4,"结果婚的和尚未结过婚的")
  )).toDF("id", "sentence")

  val seg = new JiebaSegmenter

  val znTokenizer = new ZnTokenizer().setInputCol("sentence").setOutputCol("words")

  //开始转换,
  znTokenizer.transform(sentenceDataFrame1).show(false)


}
