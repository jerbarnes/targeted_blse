package de.unibi.sc.sentiment.util;

import java.util.ArrayList;

/**
 * This class holds all read-in information that should be crawled during the
 * next step.
 *
 * @author robin
 */
public class SearchItemList {

    private ArrayList<String> internalIDs = new ArrayList();
    private ArrayList<String> amazonIDs = new ArrayList();
    private ArrayList<String> reviewIDs = new ArrayList();

    public SearchItemList() {
    }

    /**
     * If any of the three lists is empty, this will return true. Actually,
     * all three lists should have the same length, when all search items have
     * been read in.
     * @return true, if any of the internal-, amazon- or reviewID lists is
     * empty
     */
    public boolean isEmpty() {
        return internalIDs.isEmpty() || amazonIDs.isEmpty() || reviewIDs.isEmpty();
    }

    /**
     * Returns the size of the lists, if all three lists have the same size.
     * @return Size of the lists, if all three lists have the same size, -1 
     * otherwise.
     */
    public int size() {
        if (internalIDs.size() != amazonIDs.size()
                || internalIDs.size() != reviewIDs.size()
                || amazonIDs.size() != reviewIDs.size()) {
            return -1;
        }
        return internalIDs.size();
    }

    public boolean addInternalID(String iternalID) {
        return this.internalIDs.add(iternalID);
    }

    public boolean addAmazonID(String amazonID) {
        return this.amazonIDs.add(amazonID);
    }

    public boolean addReviewID(String reviewID) {
        return this.reviewIDs.add(reviewID);
    }

    public String getInternalID(int index) {
        return this.internalIDs.get(index);
    }

    public String getAmazonID(int index) {
        return this.amazonIDs.get(index);
    }

    public String getReviewID(int index) {
        return this.reviewIDs.get(index);
    }

}
