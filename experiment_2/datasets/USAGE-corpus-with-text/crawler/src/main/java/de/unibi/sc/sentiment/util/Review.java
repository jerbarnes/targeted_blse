/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package de.unibi.sc.sentiment.util;

/**
 * Helper structure, that is returned after all review information was ectracted
 * by the crawler.
 * @author robin
 */
public class Review {
    
    /**
     * Author of the review.
     */
    private String author;
    /**
     * Date of review publication.
     */
    private String date;
    /**
     * Review's title.
     */
    private String title;
    /**
     * Name of the product the review refers to.
     */
    private String productname;
    /**
     * Full review text.
     */
    private String review;
    
    public Review(){
        this.author = null;
        this.date = null;
        this.productname = null;
        this.title = null;
        this.review = null;
    }
    
    public Review(String author, String date, String productname, String title, String review){
        this.author = author;
        this.date = date;
        this.productname = productname;
        this.title = title;
        this.review = review;
    }
    
    @Override
    public String toString(){
        String ret = "### Review ###\n";
        ret += "Author: " + this.author + "\n";
        ret += "Pulicationdate: " + this.date + "\n";
        ret += "Product: " + this.productname + "\n";
        ret += "Title: " + this.title + "\n";
        ret += "Review: " + this.review + "\n\n";
        return ret;
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getProductname() {
        return productname;
    }

    public void setProductname(String productname) {
        this.productname = productname;
    }

    public String getReview() {
        return review;
    }

    public void setReview(String review) {
        this.review = review;
    }
    
}
