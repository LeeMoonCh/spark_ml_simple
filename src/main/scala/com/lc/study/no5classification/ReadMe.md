# 分类和回归

经过前面四章的学习，现在终于回归到了可以应用到线上的算法中了。分类和回归算法，主要用于给数据分类别，比如：垃圾邮件的区分，二类别分类等等。那么在本章节中，你会学习到如下分类算法：

- Classification
  - Logistic regression[逻辑回归]
    - [Binomial logistic regression](http://spark.apache.org/docs/latest/ml-classification-regression.html#binomial-logistic-regression)[二项式逻辑回归]
    - [Multinomial logistic regression[多项式逻辑回归]](http://spark.apache.org/docs/latest/ml-classification-regression.html#multinomial-logistic-regression)
  - [Decision tree classifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-classifier)[决策树]
  - [Random forest classifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier)[随机森林]
  - [Gradient-boosted tree classifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier)[梯度提高树]
  - [Multilayer perceptron classifier](http://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier)[多层感知器分类算法]
  - [Linear Support Vector Machine](http://spark.apache.org/docs/latest/ml-classification-regression.html#linear-support-vector-machine)[SVM 支持向量机]
  - [One-vs-Rest classifier (a.k.a. One-vs-All)](http://spark.apache.org/docs/latest/ml-classification-regression.html#one-vs-rest-classifier-aka-one-vs-all)
  - [Naive Bayes](http://spark.apache.org/docs/latest/ml-classification-regression.html#naive-bayes)[朴素贝叶斯]
- Regression
  - [Linear regression](http://spark.apache.org/docs/latest/ml-classification-regression.html#linear-regression)[线性回归]
  - Generalized linear regression[广义线性回归]
    - [Available families](http://spark.apache.org/docs/latest/ml-classification-regression.html#available-families)
  - [Decision tree regression](http://spark.apache.org/docs/latest/ml-classification-regression.html#decision-tree-regression)
  - [Random forest regression](http://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-regression)
  - [Gradient-boosted tree regression](http://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-regression)
  - [Survival regression](http://spark.apache.org/docs/latest/ml-classification-regression.html#survival-regression)[生存回归]
  - [Isotonic regression](http://spark.apache.org/docs/latest/ml-classification-regression.html#isotonic-regression)[保序回归]
- [Linear methods](http://spark.apache.org/docs/latest/ml-classification-regression.html#linear-methods)[线性方法]
- Decision trees[决策树相关详解]
  - Inputs and Outputs
    - [Input Columns](http://spark.apache.org/docs/latest/ml-classification-regression.html#input-columns)
    - [Output Columns](http://spark.apache.org/docs/latest/ml-classification-regression.html#output-columns)
- Tree Ensembles
  - Random Forests[随机森林相关详解]
    - Inputs and Outputs
      - [Input Columns](http://spark.apache.org/docs/latest/ml-classification-regression.html#input-columns-1)
      - [Output Columns (Predictions)](http://spark.apache.org/docs/latest/ml-classification-regression.html#output-columns-predictions)
  - Gradient-Boosted Trees (GBTs)[提高增强树相关详解]
    - Inputs and Outputs
      - [Input Columns](http://spark.apache.org/docs/latest/ml-classification-regression.html#input-columns-2)
      - [Output Columns (Predictions)](http://spark.apache.org/docs/latest/ml-classification-regression.html#output-columns-predictions-1)