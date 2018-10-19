#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [ ! -f "$DIR/node_modules/marked/bin/marked" ] ; then
	echo "marked is not installed. Install dependencies with \"npm install\"."
fi

# remove existing html pages
rm docs/*.html
rm docs/css/*
rm -rf docs/js/BBElements/*

mkdir -p docs/css
cp -r BBElements/css/* docs/css/

mkdir -p docs/js/BBElements
cp -r BBElements/js/* docs/js/BBElements/

# for each .md file in markdown (won't search recursively)
for FILE in $(ls markdown/*.md) ; do
	OUTPUT="docs/$(echo "$FILE" | sed -e "s/markdown\///" | sed -e "s/\.md$/.html/")"
	mkdir -p $(dirname "$OUTPUT")

	cat templates/header.html > "$OUTPUT"
	"$DIR/node_modules/marked/bin/marked" -i "$FILE" --gfm \
		| sed -r 's/(<a href="https{0,1}:\/\/[^>]+")>/\1 target="_blank">/' \
		>> "$OUTPUT"
	cat templates/footer.html >> "$OUTPUT"
done

# open w/ firefox
firefox docs/index.html
