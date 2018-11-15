package de.unibi.sc.sentiment.crawling;

import de.unibi.sc.sentiment.crawler.AmazonCrawler;
import de.unibi.sc.sentiment.util.Constants.TopLevelDomain;
import de.unibi.sc.sentiment.util.SearchItemList;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Factor Graph oriented Sentiment Analysis User: rklinger
 * Date: 19.03.14
 * Time: 14:44
 */
public class CrawlByAmazonID {

    public static void main(String[] args) {
        System.err.println("Amazon Review Crawler based on Review IDs");
        System.err.println("Authors: Robin Schiewer (rschiewer@techfak.uni-bielefeld.de)");
        System.err.println("and      Roman Klinger  (rklinger@cit-ec.uni-bielefeld.de)");
        checkArgs(args);
        String inputFile = args[0] ;
        String domain = args[1] ;
        String outfile = args[2] ;
        SearchItemList list = readInData(inputFile);
        checkList(list);
        start(list, domain,outfile);
    }

    /**
     * Tests the read-in list for emptiness and terminates program flow with an
     * error, if the list is empty indeed.
     *
     * @param list list to be checked
     */
    private static void checkList(SearchItemList list) {
        if (list.isEmpty()) {
            System.err.println("Error: ID list is empty");
            System.exit(1);
        }
    }

    /**
     * Checks parameters.
     *
     * @param args
     * @return Actual ini filename
     */
    private static String checkArgs(String[] args) {
        if (args.length < 3) {
            System.err.println("Please specify an input file in the following tab separated format as first parameter:");
            System.err.println("InternalID  ProductId   ReviewId");
            System.err.println("(the product ID is not actually used)");
            System.err.println("The second parameter should be the top level domain to be used.");
            System.err.println("Specify an output file as third argument.");
            System.err.println("----------------------------------------------");
            System.exit(-1);
        }
        if (!args[1].equals(TopLevelDomain.TLD_DE.getStr()) && !args[1].equals(TopLevelDomain.TLD_COM.getStr())) {
            System.err.println("Error: Please choose either 'de' or 'com' as top level domain");
            System.exit(-1);
        }
        return args[0];
    }

    /**
     * Loads the search list items.
     * @param fileName path to the search list
     * @return list with items that should be crawled or null, if there went
     * something wrong while read in the data
     */
    private static SearchItemList readInData(String fileName) {
        SearchItemList list = new SearchItemList();
        try {
            String line;
            FileReader fr = new FileReader(fileName);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                String[] fields = line.split("\t");
                assert (fields[0].length() == 4 && fields[1].length() == 10);
                list.addInternalID(fields[0]);
                list.addAmazonID(fields[1]);
                list.addReviewID(fields[2]);
            }
        } catch (FileNotFoundException ex) {
            Logger.getLogger(CrawlByAmazonID.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        } catch (IOException ex) {
            Logger.getLogger(CrawlByAmazonID.class.getName()).log(Level.SEVERE, null, ex);
            return null;
        }
        return list;
    }

    /**
     * Starts the crawl process and converts top level domain string to top
     * level domain constant.
     * @param list list of search items
     * @param tldStr top level domain string
     * @return true if crawl process went normally, false otherwise
     */
    private static boolean start(SearchItemList list, String tldStr, String outFile) {
        TopLevelDomain tld = null;
        switch (tldStr) {
            case "de":
                tld = TopLevelDomain.TLD_DE;
                break;
            case "com":
                tld = TopLevelDomain.TLD_COM;
                break;
            default:
                return false;
        }
        AmazonCrawler crawler = new AmazonCrawler(tld, 500);
        if (crawler.openStream(outFile) == true) {
            crawler.crawl(list);
            crawler.closeStream();
            return true;
        }
        return false;
    }
}
