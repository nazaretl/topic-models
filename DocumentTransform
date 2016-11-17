#!/bin/bash
# Transform documents into ASCII and remove all punctuation marks
declare -a documents=("example1" "example2" "example3")

for i in "${documents[@]}"
do
   	iconv -f ISO-8859-1 -t ASCII//TRANSLIT "$i" > "$i.txt"
	cat "$i.txt" | tr -d '[:punct:]' > "${i}c.txt"
done
