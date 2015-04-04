package org.apache.spark.mllib.ann

import java.awt._
import java.awt.event._
import java.text.SimpleDateFormat
import java.util.Calendar

import org.apache.log4j.PropertyConfigurator
import org.apache.spark._
import org.apache.spark.mllib.classification2.NeuralNetworkClassifier
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.Array.canBuildFrom
import scala.util.Random


object ANNwithLBFGS {

  var rand = new Random(0)

  def generateInput2D(f: Double => Double, xmin: Double, xmax: Double, noPoints: Int):
  Array[(Vector, Vector)] = {

    var out = new Array[(Vector, Vector)](noPoints)

    for (i <- 0 to noPoints - 1) {
      val x = xmin + rand.nextDouble() * (xmax - xmin)
      val y = f(x)
      out(i) = (Vectors.dense(x), Vectors.dense(y))
    }

    return out

  }

  def f(T: Double): Double = {
    val y = 0.5 + Math.abs(T / 5).toInt.toDouble * .15 + math.sin(T * math.Pi / 10) * .1
    assert(y <= 1)
    y
  }

  def concat(v1: Vector, v2: Vector): Vector = {

    var a1 = v1.toArray
    var a2 = v2.toArray
    var a3 = new Array[Double](a1.size + a2.size)

    for (i <- 0 to a1.size - 1) {
      a3(i) = a1(i)
    }

    for (i <- 0 to a2.size - 1) {
      a3(i + a1.size) = a2(i)
    }

    Vectors.dense(a3)

  }

  def main(arg: Array[String]) {

    println("ANN demo")

    PropertyConfigurator.configure("log4j.properties")

    val conf = new SparkConf().setAppName("Parallel ANN").setMaster("local[1]")
    val sc = new SparkContext(conf)

//    val testRDD2D =
//      sc.parallelize(generateInput2D(T => f(T), -10, 10, 100), 2).cache
//    val validationRDD2D =
//      sc.parallelize(generateInput2D(T => f(T), -10, 10, 100), 2).cache
//
//    val starttime = Calendar.getInstance().getTime()
//    println("Training")
//    val model2D = ArtificialNeuralNetwork.train(testRDD2D, Array[Int](5, 3), 3000, 1e-8)
//    val stoptime = Calendar.getInstance().getTime()
//
//    println(((stoptime.getTime - starttime.getTime + 500) / 1000) + "s")
//
//    val predictedAndTarget2D = validationRDD2D.map(T => (T._1, T._2, model2D.predict(T._1)))
//
//    var err2D = predictedAndTarget2D.map(T =>
//      (T._3.toArray(0) - T._2.toArray(0)) * (T._3.toArray(0) - T._2.toArray(0))
//    ).reduce((u, v) => u + v)
//
//    println("Error: " + err2D)


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
    val rddData = sc.parallelize(data).cache()


    val startTime = Calendar.getInstance().getTime()
    val model = ANNClassifier.train(rddData,Array(hiddenSize),2000,1,1e-4)
    val endTime = Calendar.getInstance().getTime()

    println(((endTime.getTime - startTime.getTime + 500) / 1000) + "s")

    val predictionAndLabels = rddData.map(lp => (model.predict(lp.features), lp.label)).collect()

    predictionAndLabels.foreach(x => println(x._1 , x._2))

    sc.stop

  }

}