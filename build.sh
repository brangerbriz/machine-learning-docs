#!/bin/bash

# for each .md file in markdown (won't search recursively)
for FILE in $(ls markdown/*.md) ; do
	OUTPUT="www/$(echo "$FILE" | sed -e "s/markdown\///" | sed -e "s/\.md$/.html/")"
	mkdir -p $(dirname "$OUTPUT")
	# convert to markdown, adding the "markdown-body" class for styling
	markdown "$FILE" --flavor gfm --stylesheet css/github-markdown.css | \
	sed -e "s/<body>/<body class='markdown-body'>/" | sed -e "s/<head>/<head><meta charset='utf-8'>/" > "$OUTPUT"
done

# open w/ firefox
firefox www
