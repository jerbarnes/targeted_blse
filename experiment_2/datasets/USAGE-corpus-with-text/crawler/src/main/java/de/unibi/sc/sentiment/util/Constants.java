package de.unibi.sc.sentiment.util;

/**
 * Here go all the constants used in this project, until a properties file is
 * created.
 *
 * @author robin
 */
public final class Constants {

    /**
     * Source of content obtained from Amazon.
     */
    public static final String SOURCE_AMAZON = "amazon";

    public static final String AMAZON_CRAWL_1 = "http://www.amazon.";
    public static final String AMAZON_CRAWL_2 = "/review/";

    /**
     * All top level domains as enums with appropiate top level domain Strings.
     */
    public static enum TopLevelDomain {

        TLD_DE("de"), TLD_COM("com");

        /**
         * The appropiate top level domain string.
         */
        private final String str;

        TopLevelDomain(String str) {
            this.str = str;
        }

        /**
         * Returns string for constant.
         *
         * @return The string representation as used in URLs.
         */
        public String getStr() {
            return str;
        }
    }

    /**
     * All months as constants.
     */
    public static enum Months {

        JANUARY, FEBRUARY, MARCH, APRIL, MAY, JUNE, JULY, AUGUST, SEPTEMBER,
        OCTOBER, NOVEMBER, DECEMBER;
    }

    /**
     * Private constructor to prevent instantiation.
     */
    private Constants() {
    }
}
