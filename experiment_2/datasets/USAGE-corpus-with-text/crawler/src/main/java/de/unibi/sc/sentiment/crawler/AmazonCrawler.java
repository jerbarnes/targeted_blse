/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package de.unibi.sc.sentiment.crawler;

import de.unibi.sc.sentiment.util.Constants;
import de.unibi.sc.sentiment.util.Constants.TopLevelDomain;
import de.unibi.sc.sentiment.util.LanguageSpecific;
import de.unibi.sc.sentiment.util.Review;
import de.unibi.sc.sentiment.util.SearchItemList;
import java.io.InputStream;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.apache.commons.httpclient.*;
import org.apache.commons.httpclient.methods.GetMethod;
import org.apache.commons.httpclient.params.HttpClientParams;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Date;
import org.clapper.util.html.HTMLUtil;

/**
 * Specific crawler designed to obtain product reviews from Amazon.
 *
 * @author robin
 */
public class AmazonCrawler {

    /**
     * Writer to write the crawled reviews.
     */
    private Writer fileWriter;
    /**
     * Top level domain of amazon url.
     */
    private final TopLevelDomain domain;
    /**
     * Indicates the wait time in milliseconds between two http requests.
     */
    private final int pauseTime;

    /**
     * Creates a amazon crawler with the desired properties.
     *
     * @param domain top level domain of the searcher
     * @param pauseTime break between two http requests
     */
    public AmazonCrawler(TopLevelDomain domain, int pauseTime) {
        this.domain = domain;
        this.pauseTime = pauseTime;
    }

    /**
     * Performs the actual crawl operation and extracts the reviews and
     * according metadata from amazon.
     *
     * @param list list with search items (see documentation for format)
     * @return true, if crawl approach went correct, false otherwise
     */
    public boolean crawl(SearchItemList list) {
        for (int i = 0; i < list.size(); ++i) {
            String id = list.getReviewID(i);
            System.err.println("Crawling " + id + "; number " + i);
            try {
                String url = Constants.AMAZON_CRAWL_1
                        + domain.getStr()
                        + Constants.AMAZON_CRAWL_2
                        + id;
                HttpClient client = new HttpClient();
                //setUserAgent(client, "firefox");
                HttpMethod method = new GetMethod(url);
                setEncoding(method, "text/html; charset=UTF-8");
                try {
                    int statusCode = client.executeMethod(method), time = 2;
                    while (statusCode != HttpStatus.SC_OK && time <= 16) {
                        System.err.println("Error during http request: " + statusCode + " at " + url
                                + ", retrying after " + time + " seconds...");
                        Thread.sleep(time * 1000);
                        statusCode = client.executeMethod(method);
                        time *= 2;
                    }
                    if (time == 32) {
                        System.err.println("Error during http request: " + statusCode + " at " + url
                                + ", no further retries");
                        writeDummy(list.getInternalID(i), list.getReviewID(i), list.getAmazonID(i));
                        continue;
                    }
                    String responseBody = readResponse(method);
                    Review extractedReview = extractContent(responseBody);
                    writeReview(extractedReview, list.getInternalID(i), list.getReviewID(i), list.getAmazonID(i));
                } catch (HttpException e) {
                    System.err.println("Fatal protocol violation: " + e.getMessage());
                    continue;
                } catch (IOException e) {
                    System.err.println("Fatal transport error: " + e.getMessage());
                    continue;
                } finally {
                    method.releaseConnection();
                }
                // Pause for a while to avoid ip ban
                Thread.sleep(pauseTime);
            } catch (InterruptedException e) {
                System.err.println("Could not interrupt thread: " + e.getMessage());
            } catch (IllegalArgumentException e) {
                System.err.println("Review not found: " + id);
                writeDummy(list.getInternalID(i), list.getReviewID(i), list.getAmazonID(i));
            }
        }
        return true;
    }

    /**
     * Reads the http response via a stream.
     *
     * @param method http method, with which the request was performed
     * previously
     * @return the whole answer to the request, i.e. html code of the requested
     * review page
     * @throws IOException
     */
    private String readResponse(HttpMethod method) throws IOException {
        // Read the response body via stream
        InputStream responseStream = method.getResponseBodyAsStream();
        String encoding = method.getResponseHeader("Content-Type").getValue().split("=")[1];
        Scanner scanner = new Scanner(responseStream, encoding);
        String responseBody = "";
        while (scanner.hasNextLine()) {
            responseBody += scanner.nextLine() + "\n";
        }
        return responseBody;
    }

    /**
     * Sets encoding of the http request explicitly.
     *
     * @param method method of which encoding should be set
     * @param encoding desired enciding
     */
    private void setEncoding(HttpMethod method, String encoding) {
        method.setRequestHeader("Content-Type", encoding);
    }

    /**
     * Sets the user agent explicitly.
     *
     * @param client client for http request
     * @param userAgent desired user agent (e.g. "firefox")
     */
    private void setUserAgent(HttpClient client, String userAgent) {
        HttpClientParams params = client.getParams();
        System.out.println("User agent is:" + params.getParameter("http.useragent"));
        params.setParameter("http.useragent", userAgent);
    }

    /**
     * Extracts review and metadata from raw html string.
     *
     * @param input raw html string
     * @return Review object with extracted review data
     */
    private Review extractContent(String input) {
        String date = null, author = null, productname = null, title = null, review = null;
        Pattern datePattern = Pattern.compile(".*<nobr>(.*)</nobr>.*</div>.*<div style=\"margin-bottom:0.5em;\">.*", Pattern.DOTALL);
        Pattern authorPattern = Pattern.compile(".*<div><div style=\"float:left;\">.*</div><div style=\"float:left;\"><a href=.* ><span style = \"font-weight: bold;\">(.*)</span></a></div></div><div style=\"clear:both;\"></div>.*", Pattern.DOTALL);
        Pattern productnamePattern = Pattern.compile(".*<div class=\"tiny\" style=\"margin-bottom:0.5em;\">.*?<b><span class=\"h3color tiny\">.*?</span>(.*?)</b>.*?</div>.*", Pattern.DOTALL);
        Pattern titlePattern = Pattern.compile(".*</span>.*?<b>(.*?)</b>, <nobr>.*", Pattern.DOTALL);
        Pattern reviewPattern = Pattern.compile(".*<div class=\"reviewText\">(.*?)</div>.*", Pattern.DOTALL);

        Matcher m = datePattern.matcher(input);
        date = m.matches() ? m.group(1) : "no match found";

        m = authorPattern.matcher(input);
        author = m.matches() ? m.group(1) : "no match found";

        m = productnamePattern.matcher(input);
        productname = m.matches() ? m.group(1) : "no match found";

        m = titlePattern.matcher(input);
        title = m.matches() ? m.group(1) : "no match found";

        m = reviewPattern.matcher(input);
        review = m.matches() ? m.group(1) : "no match found";

        String cleanedReviewTitle = HTMLUtil.textFromHTML(title).replaceAll("&#34;","\"").replaceAll("&#252;","&uuml;") ;
        String cleanedReviewText = HTMLUtil.textFromHTML(review).replaceAll("&#34;","\"").replaceAll("&#252;","&uuml;") ;

        return new Review(author, transformDate(date), productname, cleanedReviewTitle, cleanedReviewText); // HTMUtil removes e.g. <br> tags from review
    }

    /**
     * Opens the output stream to the specified path.
     *
     * @param path path to the output file (filename itself inclusive)
     * @return true if stream was opened without problems, false otherwise
     */
    public boolean openStream(String path) {
        try {
            FileOutputStream fos = new FileOutputStream(path, false);
            fileWriter = new BufferedWriter(new OutputStreamWriter(fos, "UTF-8"));
            return true;
        } catch (IOException ex) {
            System.err.println("Could not open stream to print crawled reviews: " + ex);
            return false;
        }
    }

    /**
     * Appends another review to the output file. Output files with a given name
     * are always overwritten if already present before the start of the crawl
     * process.
     *
     * @param review the review to be written
     * @param internalID internal id of the crawled review
     * @param reviewID amazon review id of the review
     * @param amazonID amazon id of the product the review refers to
     * @return true if the write process went okay, false otherwise
     */
    private boolean writeReview(Review review, String internalID, String reviewID, String amazonID) {
        try {
            fileWriter.append(internalID + "\t");
            fileWriter.append(amazonID + "\t");
            fileWriter.append(reviewID + "\t");
            fileWriter.append(replaceSpecChars(review.getProductname()) + "\t");
            fileWriter.append(replaceSpecChars(review.getTitle()) + "\t");
            //fileWriter.append(replaceSpecChars(review.getAuthor()) + "\t");
            //fileWriter.append(review.getDate() + "\t");
            fileWriter.append(replaceSpecChars(review.getReview()) + "\n");
            return true;
        } catch (IOException ex) {
            return false;
        }
    }

    /**
     * Used to write all input IDs in the output file, when no review could be
     * extracted.
     *
     * @param internalID internal id of the crawled review
     * @param reviewID amazon review id of the review
     * @param amazonID amazon id of the product the review refers to
     * @return true if the write process went okay, false otherwise
     */
    private boolean writeDummy(String internalID, String reviewID, String amazonID) {
        try {
            fileWriter.append(internalID + "\t");
            fileWriter.append(amazonID + "\t");
            fileWriter.append(reviewID + "\t");
            fileWriter.append("[DISAPPEARED]" + "\n");
            return true;
        } catch (IOException ex) {
            return false;
        }
    }

    /**
     * Closes the write stream after the crawl process finished successfully.
     *
     * @return true if stream could be closed successfully, false otherwise
     */
    public boolean closeStream() {
        try {
            fileWriter.close();
            return true;
        } catch (IOException ex) {
            System.err.println("Could not close stream: " + ex);
            return false;
        }
    }

    /**
     * Transforms the Amazon given date format to DD.MM.YYYY numeric.
     *
     * @param date input date of the format DD.<month>.YYYY
     * @return reformatted string of format DD.MM.YYYY
     */
    private String transformDate(String date) {
        String temp[] = date.split(" ");

        String day = "";
        String month = "";
        String year = "";

        // Date will be formatted equally, regardless of the language
        if (domain.equals(TopLevelDomain.TLD_DE)) {
            day = temp[0];
            month = temp[1];
            year = temp[2];
        } else if (domain.equals(TopLevelDomain.TLD_COM)) {
            day = temp[1];
            month = temp[0];
            year = temp[2];
        } else {
            System.err.println("Invalid top level domain, date will contain errors!");
            day = "err";
            month = "err";
            year = "err";
        }

        if (day.length() == 2) {
            day = "0" + day;
        }
        day = day.replace(",", ".");

        month = HTMLUtil.textFromHTML(month);
        month = convertMonth(month);

        return day + month + year;
    }

    /**
     * Converts given month string to a numeric value. Uses top level domain for
     * appropriate string matching. Currently, only German and English are
     * supported.
     *
     * @param month String which should be converted
     * @return String with numeric month value
     */
    private String convertMonth(String month) {
        if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.JANUARY, domain))) {
            month = "01.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.FEBRUARY, domain))) {
            month = "02.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.MARCH, domain))) {
            month = "03.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.APRIL, domain))) {
            month = "04.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.MAY, domain))) {
            month = "05.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.JUNE, domain))) {
            month = "06.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.JULY, domain))) {
            month = "07.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.AUGUST, domain))) {
            month = "08.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.SEPTEMBER, domain))) {
            month = "09.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.OCTOBER, domain))) {
            month = "10.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.NOVEMBER, domain))) {
            month = "11.";
        } else if (month.equals(LanguageSpecific.
                getMonth(Constants.Months.DECEMBER, domain))) {
            month = "12.";
        } else {
            System.err.println("Unknown month: " + month + ", can not convert");
            month = "err";
        }
        return month;
    }

    /**
     * Replaces special html characters with appropiate utf-8 encoded
     * characters.
     *
     * @param input input with html characters
     * @return cleaned input without html characters
     */
    private String replaceSpecChars(String input) {
        input = input.replace("<br />", "\n");
        input = input.replace("&auml;", "ä");
        input = input.replace("&ouml;", "ö");
        input = input.replace("&uuml;", "ü");
        input = input.replace("&Auml;", "Ä");
        input = input.replace("&Ouml;", "Ö");
        input = input.replace("&Uuml;", "Ü");
        input = input.replace("&quot;", "\"");
        input = input.replace("&amp;", "&");
        input = input.replace("&szlig;", "ß");
        return input;
    }
}
