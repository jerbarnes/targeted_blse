package de.unibi.sc.sentiment.util;


import de.unibi.sc.sentiment.util.Constants.Months;
import de.unibi.sc.sentiment.util.Constants.TopLevelDomain;
import java.lang.reflect.Field;

/**
 * Here are stored language specific regular expressions.
 *
 * @author robin
 */
public class LanguageSpecific {

    /**
     * The review rating part specific for german top level domain.
     */
    private static final String AMAZON_REVIEW_RATING_DE = ".*?([0-9]*)\\svon\\s([0-9]*)\\sKunden.*";
    /**
     * The review rating part specific fot english top level domain.
     */
    private static final String AMAZON_REVIEW_RATING_COM = ".*?([0-9]*)\\sof\\s([0-9]*)\\speople.*";
    /**
     * All months for various top level domains (which are in different
     * languages).
     */
    private static final String JANUARY_DE = "Januar";
    private static final String JANUARY_COM = "January";
    private static final String FEBRUARY_DE = "Februar";
    private static final String FEBRUARY_COM = "February";
    private static final String MARCH_DE = "März";
    private static final String MARCH_COM = "March";
    private static final String APRIL_DE = "April";
    private static final String APRIL_COM = "April";
    private static final String MAY_DE = "Mai";
    private static final String MAY_COM = "May";
    private static final String JUNE_DE = "Juni";
    private static final String JUNE_COM = "June";
    private static final String JULY_DE = "Juli";
    private static final String JULY_COM = "July";
    private static final String AUGUST_DE = "August";
    private static final String AUGUST_COM = "August";
    private static final String SEPTEMBER_DE = "September";
    private static final String SEPTEMBER_COM = "September";
    private static final String OCTOBER_DE = "Oktober";
    private static final String OCTOBER_COM = "October";
    private static final String NOVEMBER_DE = "November";
    private static final String NOVEMBER_COM = "November";
    private static final String DECEMBER_DE = "Dezember";
    private static final String DECEMBER_COM = "December";
    
    /**
     * AMAZON: Returns the appropiate language-specific regular expression part
     * for the extraction of amazon review ratings.
     *
     * @param topLevelDomain one of the allowed top level domains from Constants
     * class
     * @return the needed regex part depending on the top level domain or null,
     * if top level domain is unknown
     */
    public static String getAmazonReviewRating(TopLevelDomain topLevelDomain) {
        if (topLevelDomain.equals(TopLevelDomain.TLD_DE)) {
            return AMAZON_REVIEW_RATING_DE;
        } else if (topLevelDomain.equals(TopLevelDomain.TLD_COM)) {
            return AMAZON_REVIEW_RATING_COM;
        }
        System.err.println("LanguageSpecific::getAmazonReviewRating: Unknown top level domain!");
        return null;
    }

    /**
     * AMAZON: Returns the appropiate language-specific regular expression part
     * for the conversion to numeric expressions of months in dates.
     * 
     * @param month the month, which's string should be obtained 
     * @param topLevelDomain the associated top level domain
     * @return the month string depending on the top level domain and the
     * desired month
     */
    public static String getMonth(Months month, TopLevelDomain topLevelDomain) {
        String fieldName = "";
        try {
            fieldName = month.toString().toUpperCase() + "_" + topLevelDomain.getStr().toUpperCase();
            Field field = LanguageSpecific.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(null).toString();
        } catch (NoSuchFieldException ex) {
            System.err.println("LanguageSepcific::getMonth: No souch field found \"" + fieldName + "\"!");
        } catch (IllegalAccessException ex) {
            System.err.println("LanguageSepcific::getMonth: No access to field granted!");
        }
        return null;
    }
}
