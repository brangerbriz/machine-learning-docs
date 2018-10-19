# Machine Learning Docs

This repository contains text, source code, and build tools for the open source *Machine Learning Documentation & Guides* project at [Branger_Briz](https://brangerbriz.com/). The goal of this project is to provide free (libre) documentation, tutorials, and guides for developers at Branger_Briz & elsewhere to learn about contemporary machine learning techniques and practices, with hopes that they can include these techniques in their own software development projects. The project was authored from 2017 to 2018 by [Brannon Dorsey](https://twitter.com/brannondorsey) and is released under the terms of the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) and [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) creative commons licenses (for source code and text respectively).

> **UPDATE** (Oct 19, 2018): This project is under active development and should be considered a draft release.

## For Developers & Contributors

### Install

```bash
# clone the repository and the necessary submodules
git clone --recursive git@brangerbriz.com:bdorsey/machine-learning-docs
cd machine-learning-docs

# Install dependencies
npm install
```

### Build

To build the `www/*.html` website files from the `markdown/*.md` source files, run `build.sh`. If you have Firefox installed, a local copy of the website will also be opened in your browser.

```bash
# rebuild the site after edits to files in markdown/
./build.sh
```