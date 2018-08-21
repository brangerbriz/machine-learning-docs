## Install

```bash
# clone
git clone git@brangerbriz.com:bdorsey/machine-learning-docs
cd machine-learning-docs

# Download the BBElements submodule
git submodule init && git submodule update

# Install dependencies
npm install
```

## Build

To build the `www/*.html` website files from the `markdown/*.md` source files, run `build.sh`.

```bash
# rebuild the site after edits to files in markdown/
./build.sh
```