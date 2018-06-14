import scala.io.Source

object MovAvg {
  def main(args: Array[String]): Unit = {
    val numbers = Source.stdin.getLines().flatMap(_.split("\\s")).map(_.toDouble).toList

    val length = numbers.length.toDouble
    val mean = numbers.sum / length
    val stdDev = Math.sqrt((numbers.map(_ - mean)
      .map(t => t * t).sum) / (length))

    println(f"$stdDev%1.5f")
  }
}