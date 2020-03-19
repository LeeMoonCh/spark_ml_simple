package com.lc.study.no1Statistics

import org.apache.spark.ml.linalg.{Vectors,Vector}
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.sql.SparkSession

/**
 * 统计学中的假设检测。先上相关术语解释（@以下内容引自百度百科）：
 * 假设检验(hypothesis testing)，又称统计假设检验，
 * 是用来判断样本与样本、样本与总体的差异是由抽样误差引起还是本质差别造成的统计推断方法。
 *
 * @以下内容  https://baijiahao.baidu.com/s?id=1629858003935285309&wfr=spider&for=pc
 * 假设检验是数理统计学中根据一定假设条件由样本推断总体的一种方法。事先对总体参数或分布形式作出某种假设，
 * 然后利用样本信息来判断原假设是否成立，采用逻辑上的反证法，依据统计上的小概率原理。
 *
 * 该网站上举了一个100豆，猜红豆和黑豆是否一样。  需要注意的是 当假设检测结果小于0.05时，可以反证假设为
 * @无效假设。
 *
 * 在我们spark中，使用类ChiSquareTest 进行假设检测。
 * 但是在我们的spark中，它的假设检测是使用的皮尔森卡方检测。
 * @以下引用知乎：https://www.zhihu.com/question/325582318/answer/693733653
 *
 * 皮尔逊卡方检验是检验实际频数和理论频数是否较为接近，
 * 检验统计量是：X^2=∑{【（实际频数-理论频数的）^2】/理论频数}
 * 它近似服从自由度为V =组格数－估计参数个数－1 的 分布。式中，
 * n 是样本量， 理论频数是由样本量乘以由理论分布确定的组格概率计算的。
 * 求和项数为组格数目。
 * 皮尔逊 卡方 统计量的直观意义十分显然： 是各组格的实际观测频数与理论期望频数
 * 的相对平方偏差的总和，若值充分大，则应认为样本提供了理论分布与统计分布不同的显著证据，
 * 即假设的总体分布与总体的实际分布不符，从而应否定所假定的理论分布
 *
 * 我们来看例子
 */

object HypothesistestingDemo extends App{

  //在spark中，使用假设检测时，一定要保证我们的数据集具有特征向量以及该特征向量的标签。
  //术语解释：
  //特征向量：是将某个计算需求的样本 进行向量话，如我们判断一个人是否肥胖，那么我们必须知道该人的
  //体重、身高。此时：我们就有特征向量 x = (180,70) 其中180是身高，70为体重。
  //标签：对就是打标签的标签。比如上面的肥胖，我们有两个标签值，一个胖，一个瘦，此时我们可以把胖定义为0
  //瘦定义为1，那么我们就会有：[1,(180,70)]  -> 这样一个数据样本。
  val data = Seq(
    (0.0, Vectors.dense(0.5, 10.0)),
    (0.0, Vectors.dense(1.5, 20.0)),
    (1.0, Vectors.dense(1.5, 30.0)),
    (0.0, Vectors.dense(3.5, 30.0)),
    (0.0, Vectors.dense(3.5, 40.0)),
    (1.0, Vectors.dense(3.5, 40.0))
  ) //定义数据，包含了label 以及特征向量。
  val spark = SparkSession.builder().master("local[*]").getOrCreate()
  spark.sparkContext.setLogLevel("warn")

  import spark.implicits._
  val df = data.toDF("label", "features")  //将数据集变为df，有两列。

  df.show()//看一下数据集。
//  +-----+----------+
//  |label|  features|
//  +-----+----------+
//  |  0.0|[0.5,10.0]|
//  |  0.0|[1.5,20.0]|
//  |  1.0|[1.5,30.0]|
//  |  0.0|[3.5,30.0]|
//  |  0.0|[3.5,40.0]|
//  |  1.0|[3.5,40.0]|
//  +-----+----------+


  //进行假设检验，使用皮尔森卡方检测。拿到结果集的第一行。
  val chi = ChiSquareTest.test(df, "features", "label").head

  println(chi.toSeq.mkString("//")) //看一下chi的结果。

  //皮尔森卡方结果
  println(s"pValues = ${chi.getAs[Vector](0)}")
  //自由度计算
  println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
  //统计结果。
  println(s"statistics ${chi.getAs[Vector](2)}")

  //以上是官方给出的代码。如果没有深入研究过卡方检测，根本看不懂。就像我。所以，找了一篇文章，
  //用代码写了一下他举的例子。
  /*
  @以下引自：https://www.it610.com/article/5073760.htm

  假设某工厂正在生产一批图钉，之前的均值是：1.500，标准差：0.015，下面是一组机器生产的30个图钉

  A:1.497 1.506 1.518 1.524 1.498 1.511 1.520 1.515 1.512 1.511
  B:1.498 1.516 1.528 1.514 1.488 1.521 1.521 1.525 1.513 1.521
  C:1.498 1.513 1.522 1.524 1.498 1.521 1.511 1.524 1.523 1.521

  问：该机子是否正常。（接近1为正结果，接近0位负结果。）
  我们有样本数据30个，并且，可以根据标准差进行标签标记。故，可以获取到一个
  （标签，特征）向量的数据集。
   */

  def getSeq(data:Double*):Seq[(Int,Vector)]={
    data.map(f=>{
      if( f>=(1.5-0.015) && f<=(1.5+0.015)){ //在正常值范围内
        (1,Vectors.dense(f))
      }else{
        (0,Vectors.dense(f)) //不在正常值范围内。
      }
    })
  }


  val df1 = getSeq(1.497 ,1.506 ,1.518 ,1.524 ,1.498, 1.511 ,1.520 ,1.515 ,1.512 ,1.511,
    1.498 ,1.516, 1.528 ,1.514, 1.488 ,1.521 ,1.521, 1.525 ,1.513, 1.521,
    1.498, 1.513, 1.522, 1.524, 1.498, 1.521, 1.511, 1.524, 1.523, 1.521
  ).toDF("l","v")

  val chi1 = ChiSquareTest.test(df1, "v", "l").head

  println("----")
  //皮尔森卡方结果
  //结果的pvalue=0.02634507828353605  所以不接受假设，机器是坏的。
  println(s"pValues = ${chi1.getAs[Vector](0)}")
  //自由度计算。
  /*
  自由度(degree of freedom, df)在数学中能够自由取值的变量个数，如有3个变量x、y、z，
  但x+y+z=18，因此其自由度等于2。在统计学中，自由度指的是计算某一统计量时，取值不受限制的变量个数。
  通常df=n-k。其中n为样本含量，
  k为被限制的条件数或变量个数，或计算某一统计量时用到其它独立统计量的个数。自由度通常用于抽样分布中。
   */
  println(s"degreesOfFreedom ${chi1.getSeq[Int](1).mkString("[", ",", "]")}")
  //统计结果。
  println(s"statistics ${chi1.getAs[Vector](2)}")



}
