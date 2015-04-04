package org.apache.spark.mllib.classification2

/**
 * Created by Zehao on 2015/4/4.
 */

import java.util.Calendar

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.mllib.ann.ANNClassifier
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import scala.math.random
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint



object NeuralNetworkSuite {


  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("log4j.properties")

    val conf = new SparkConf().setAppName("NeuralNetworkXORTest").setMaster("local[1]")
    val sc = new SparkContext(conf)
    /* training set */
    val inputs = Array[Array[Double]](
      Array[Double](0,0),
      Array[Double](0,1),
      Array[Double](1,0),
      Array[Double](1,1)
    )
    val outputs = Array[Double](1, 2, 3, 4)
    /* NN */
    val inputSize = 2
    val hiddenSize = 5
    val outputSize = 1
    val data = inputs.zip(outputs).map{ case(features, label) =>
      new LabeledPoint(label, Vectors.dense(features))}

//    val rddData = sc.parallelize(data, 2)
    val rddData = sc.parallelize(data)

    val startTime = Calendar.getInstance().getTime()
    val predictor = NeuralNetworkClassifier.train(rddData, Array(hiddenSize), 100, 0.3)
    val endTime = Calendar.getInstance().getTime()

    println(((endTime.getTime - startTime.getTime + 500) / 1000) + "s")


    val predictionAndLabels = rddData.map(lp => (predictor.predict(lp.features), lp.label)).collect()

    predictionAndLabels.foreach(x => println(x._1 , x._2))
//    predictionAndLabels.foreach(x => assert(x._1 == x._2))
  }

}