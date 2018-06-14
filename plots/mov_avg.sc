import scala.io.Source

object MovAvg {
  def main(args: Array[String]): Unit = {
    val range = args(0).toInt
    val numbers = Source.stdin.getLines().flatMap(_.split("\\s")).map(_.toDouble)

    val movingAverages = numbers
      .sliding(range)
      .map(_.sum)
      .map(_ / range)

    for(average <- movingAverages) println(average)
  }
}