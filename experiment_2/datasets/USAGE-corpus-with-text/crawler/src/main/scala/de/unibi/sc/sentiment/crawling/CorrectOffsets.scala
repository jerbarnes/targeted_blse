package de.unibi.sc.sentiment.crawling

import collection.mutable.{ArrayBuffer, HashMap}
import io.Source


/**
 * Factor Graph oriented Sentiment Analysis
 * User: rklinger
 * Date: 19.03.14
 * Time: 15:17
 */
object Entry {
  def apply(line:String) = {
    val s = line.split("\t")
    new Entry(s(0).trim,s(1).trim,s(2).toInt,s(3).toInt,s(4).trim,s(5).trim,s(6).trim,s(7).trim)
  }
  def apply() = new Entry("","",-1,-1,"","","","")
}
object TxtEntry {
  def apply(line:String) = {
    val s = line.split("\t")
    new TxtEntry(s(0),s(1),s(2),s(3),s(4),s(5))
  }
}
/*
2236    B000ALVUM6      143     Philips HD7546/20 Thermo Kaffeemaschine (1000 W, Tropf-Stopp Funktion) schwarz/Metall   Preis Leistung Weltklasse       Die Kaffeemaschine wurde für das Büro angeschafft. Seit geraumer Zeit ist das gerät im Dauereinsatz ( ca 8 Brühvorgänge ( 8 Liter ) pro Tag.)Bislang keine Mängel erkennbar. Entgegen der AMAZON beschreibung, schaltet sich die Maschine auch von selber aus.das H
 */
case class Entry(classs:String, internalId:String, leftOffset:Int, rightOffset:Int, stringRepr:String, annotationId:String, foreigness:String, relatedness:String) {
  override def toString() = classs+"\t"+internalId+"\t"+leftOffset+"\t"+rightOffset+"\t"+stringRepr+"\t"+annotationId+"\t"+foreigness+"\t"+relatedness
  def matches(text:String) = {
     rightOffset <= text.length && text.substring(leftOffset,rightOffset) == stringRepr
  }
  def isNonNull = { leftOffset != -1 }
  def toString(txt:String) = {
    val newStringRepr = if (rightOffset < txt.length) txt.substring(leftOffset,rightOffset) else stringRepr+"[OUT-OF-LENGTH]"
    classs+"\t"+internalId+"\t"+leftOffset+"\t"+rightOffset+"\t"+newStringRepr+"\t"+annotationId+"\t"+foreigness+"\t"+relatedness
  }
}
case class TxtEntry(internalId:String, productId:String, reviewId:String, name:String, reviewTitle:String, reviewText:String)
object CorrectOffsets {
  final val OFFSET = 4
  def main(args:Array[String]) : Unit = {
    System.err.println("USAGE Corpus Offset correction") ;
    System.err.println("Author: )") ;
    System.err.println("Roman Klinger  (rklinger@cit-ec.uni-bielefeld.de)") ;
    System.err.println("This program corrects possible offset errors based on encoding issues when crawling reviews.")
    if (args.length < 2) {
      System.err.println("Please specify an input text file and the according CSV file with offset annotations.")
      System.err.println("The text file should have the format:");
      System.err.println("InternalId    ProductId      InternalReviewID      ProductTitle   ReviewTitelAndText");
      System.err.println("The offsets are counted from the 5th column on.");
      System.err.println("----------------------------------------------");
      System.exit(-1);
    }
    start(args(0),args(1))
  }
  def start(txtFile:String,csvFile:String) {
    System.err.println("Reading text file into memory: "+txtFile+" ...")
    val internalIdText = readTxtFile(txtFile)
    // check each line in txt file
    for (line <- Source.fromFile(csvFile).getLines()) {
      val csvEntry = Entry(line)
      val text = internalIdText(csvEntry.internalId)
      if (csvEntry.matches(text)) {
        println(csvEntry)
//        System.err.println(line+" OK!")
//        Thread.sleep(100)
      }
      else {
        val correctedEntry = correctEntry(csvEntry,text)
        if (correctedEntry.isNonNull) {
          System.err.println("Correcting:")
          System.err.println(csvEntry.toString(text))
          System.err.println(correctedEntry.toString(text))
          println(correctedEntry)
        } else {
          System.err.println("Non correctable:")
          System.err.println(csvEntry.toString(text))
        }
        //Thread.sleep(10000)
      }

    }
    System.err.println("Done.")
  }
  def correctEntry(csvEntry:Entry,txt:String) : Entry = {
    if (csvEntry.matches(txt)) csvEntry
    else {
      val indices = allStartIndices(csvEntry,txt)
      if (indices.nonEmpty) {
        val betterLeftOffset = indices.head
        val betterRightOffset = indices.head + csvEntry.stringRepr.length
        Entry(csvEntry.classs,csvEntry.internalId,betterLeftOffset,betterRightOffset,csvEntry.stringRepr,csvEntry.annotationId,csvEntry.foreigness,csvEntry.relatedness)
      } else {
        Entry()
      }
    }
  }
  def readTxtFile(txtFile:String) : HashMap[String,String] = {
    val s = Source.fromFile(txtFile)
    val internalIdToTxtEntry = new HashMap[String,String]
    for (line <- s.getLines()) {
      val entry = TxtEntry(line)
      internalIdToTxtEntry += entry.internalId -> (entry.reviewTitle+" "+entry.reviewText)
    }
    internalIdToTxtEntry
  }

  /**
   * Returns all occurrences sorted by distance to original offset. Probably the first is good. It may return an empty list as well, if nothing is found!
   * @param entry
   * @param textt
   * @param startOffset
   * @param maxEndOffset
   * @return
   */
  def allStartIndices(entry:Entry,textt:String,startOffset:Int=0,maxEndOffset:Int=Int.MaxValue) : Seq[Int] = {
    val pattern = entry.stringRepr
    val endOffset = scala.math.min(textt.length,maxEndOffset)
//    if (pattern.contains("zuempfehl")) {
//      System.err.println("=======================================")
//      System.err.println(textt)
//      System.err.println(entry)
//      System.err.println(endOffset)
//      System.err.println(pattern)
//      System.err.println(textt.indexOf(pattern))
//      System.err.println("=======================================")
//    }
    val originalLeftOffset = entry.leftOffset
//    System.err.println("Searching for "+pattern+" in "+textt+" starting at "+startOffset)

    startOffset.until(endOffset).filter(textt.startsWith(pattern, _)).sortBy(x => scala.math.abs(originalLeftOffset-x))
  }
}
