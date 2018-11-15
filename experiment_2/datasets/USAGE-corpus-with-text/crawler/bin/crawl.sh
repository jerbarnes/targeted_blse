#!/bin/bash
here=`dirname $0`
targetPath=${here}/../target/recrawling-0.5-jar-with-dependencies.jar 
java -Xmx2g -jar ${targetPath} "$@"
