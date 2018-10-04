# Twitterbot Part 4: Transfer Learning, Fine-tuning, and User Personalization

In [Part 3](twitterbot-part-3-model-inference-and-deployment.html) of this tutorial we used seven million tweets to train a base model in Keras, which we then deployed in a browser environment using Tensorflow.js. In this chapter, we'll learn how we can use a technique called *transfer learning* to fine-tune our base model using tweets from individual twitter accounts. We'll create a graphical application that allows you to train models using an individual user's tweets and use them to generate synthetic tweets in the style of that user.

## Individual Twitter User Data

Until now, we've been using twitter data aggregated from hundreds of thousands of different twitter accounts. This has worked well for the purpose of training a base model to synthesize general tweets, but now we'd like to imitate individual twitter user accounts. To do so, we'll use the Twitter API. We'll create a Node.js server<span class="marginal-note" data-info="We'll be recreating the code from this [tweet-server](https://github.com/brangerbriz/tweet-server) repository, if you care to jump ahead."></span> that we can use to download a user's public tweets given their username. We'll then use this server process to download twitter data at will upon request from our training code.

<pre class="code">
    <code class="bash" data-wrap="false">
# leave the tfjs-tweet-generation directory to create and enter tweet-server/
cd ..
mkdir tweet-server
cd tweet-server/
    </code>
</pre>

Next create a `package.json` file inside of `tweet-server/` and populate it using the contents below.

<pre class="code">
    <code class="json" data-wrap="false">
{
  "name": "tweet-server",
  "version": "0.1.0",
  "description": "Download twitter data using an HTTP REST API.",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "author": "Brannon Dorsey + Nick Briz",
  "license": "GPL-3.0",
  "dependencies": {
    "cors": "^2.8.4",
    "express": "^4.16.3",
    "twit": "^2.2.11"
  }
}
    </code>
</pre>

Install the project dependencies using NPM.

<pre class="code">
    <code class="bash" data-wrap="false">
npm install
    </code>
</pre>

Before you can use the Twitter API, you have to create a Twitter API application using a developer account at [developer.twitter.com](https://developer.twitter.com). You'll need to submit an application to become a developer as well as to create a new application.<span class="marginal-note" data-info="Both applications are instantly approved in my experience, but in theory, the process can take longer."></span> Once that's done you'll need to generate consumer API keys and access tokens.

<section class="media" data-fullwidth="false">
    <img src="images/twitter-api.png"> 
</section>

We'll use Express to create our own REST API using Node.js. We'll download a user's tweets using GET requests by providing a twitter username in the URL. A request like `http://localhost:3000/api/barackobama` will return a JSON object containing Obama's tweets.

Create a new file called `server.js` and fill it with the contents below. Replace the `TWITTER_*` constants using values from your Twitter API app.

<pre class="code">
    <code class="javascript" data-wrap="false">
const path = require('path')
const cors = require('cors')
const Twit = require('twit')
const express = require('express')
const app = express()
const http = require('http').Server(app)

// populate these constants using the API keys from your Twitter API app
const TWITTER_CONSUMER_KEY=""
const TWITTER_CONSUMER_SECRET=""
const TWITTER_ACCESS_TOKEN=""
const TWITTER_ACCESS_TOKEN_SECRET=""

// create an instance of Twit, which we'll use to access the twitter API
const T = new Twit({
    // goto: https://apps.twitter.com/ for keys
    consumer_key: TWITTER_CONSUMER_KEY,
    consumer_secret: TWITTER_CONSUMER_SECRET,
    access_token: TWITTER_ACCESS_TOKEN,
    access_token_secret: TWITTER_ACCESS_TOKEN_SECRET,
    timeout: 60 * 1000 // optional HTTP request timeout for requests
})

// allows Cross-Origin Resource Sharing (CORS) on the /api endpoint
app.use('/api', cors())

// all GET requests to /api/ should include a user value in the path. 
// This user value will be interpreted as twitter handle.
app.get('/api/:user', async (req, res) => {
    const user = req.params.user
    try {
        const tweets = await getUserTweets(user)
        console.log(`[server] /api/:user got tweets for user ${user}`)
        res.json({ error: null, tweets: tweets })
    } catch (err) {
        console.log(`[server] /api/:user got tweets for user ${user} with error:`)
        console.error(err)

        let message = `Error fetching tweets for user ${user}`
        if (err.statusCode) {
            res.status(err.statusCode)
        } else {
            res.status(500)
        }

        if (err.message) message = err.message
        res.json({ error: message, tweets: null })
    }
})

http.listen(3000, () => { 
    console.log('[server] Listening on http://0.0.0.0:3000')
})

// download ~3,200 of the user's most recent tweets.
async function getUserTweets(user) {
    // https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline.html

    const tweets = []
    let batch = await getUserTweetBatch(user)

    tweets.push(...batch)
    console.log(`[twitter] Got ${batch.length} new tweets. Total ${tweets.length}`)
    // the twitter API we're using only allows tweets to be downloaded in groups
    // of 200 per request, so we create a loop to download tweets in batches
    while (batch.length > 1) {
        // use tweet ids for pagination
        let id = batch[batch.length - 1].id
        batch = await getUserTweetBatch(user, id)
        tweets.push(...batch)
        console.log(`[twitter] Got ${batch.length} new tweets. Total ${tweets.length}`)
    }
    // discard metadata and only return the contents of the tweets
    return tweets.map(tweet => tweet.text)
}

// download a batch of 200 tweets using maxId for pagination
function getUserTweetBatch(user, maxId) {
    return new Promise((resolve, reject) => {
        T.get('statuses/user_timeline', {
            screen_name: user,
            count: 200, // max (but can be called again with max_id)
            max_id: maxId,
            include_rts: true
        }, (err, data, res) => {
            if (err) reject(err)
            else resolve(data)
        })
    })
}
    </code>
</pre>

In this script, we register an HTTP route for all `GET /api/:user` requests, where `:user` is a twitter handle. All requests that match this route will trigger a download of ~3,200 tweets using the [twit npm package](https://www.npmjs.com/package/twit) via `getUserTweets()`. This function downloads 200 tweets at a time in a loop using `getUserTweetBatch()`. Batches of tweets are combined into one array and stripped of metadata; the value returned from `getUserTweets()` is an array of tweet contents only. If the download succeeded a JSON object containing the tweets is returned as the result of the HTTP request: `{ error: null, tweets: [...] })`. If there was an error downloading tweets, the tweets object is null: `{ "error": "Sorry, that page does not exist.", "tweets": null }`.

<pre class="code">
    <code class="bash" data-wrap="false">
node server.js
    </code>
</pre>

Let's query our new API in another terminal window. 

<pre class="code">
    <code class="bash" data-wrap="false">
# query the tweet-server using curl. This should take a few seconds and then
# print Obama's tweets as a JSON object like the one below.
curl http://localhost:3000/api/barackobama
    </code>
</pre>
<pre class="code">
    <code class="json" data-wrap="false">
{
    "error":null,
    "tweets": [ 
        "Today, I’m proud to endorse even more Democratic candidates who aren’t just running against something, but for some… https://t.co/oqewS0Y8vZ",
        "From civil servants to organizers, the young people I met in Amsterdam today are doing the hard work of change. And… https://t.co/mlAp2SRZlP",
        "The antidote to government by a powerful few is government by the organized, energized many. This National Voter Re… https://t.co/3W5pfaUdKd",
        "The first class of Obama Fellows is full of leaders like Keith—hardworking, innovative, and dedicated to partnering… https://t.co/nOd6FzH23n",
        "We will always remember everyone we lost on 9/11, thank the first responders who keep us safe, and honor all who de… https://t.co/ku270JQnwl",
        "RT @nowthisnews: If you still don't think the midterms will affect you, @BarackObama is back to spell out just how important they are https…",
        "Today I’m at the University of Illinois to deliver a simple message to young people all over the country: You need… https://t.co/brM6Vd7j2R",
        "Yesterday I met with high school students on Chicago’s Southwest side who spent the summer learning to code some pr… https://t.co/hY9B0mSQB9",
        "Congratulations to Hawaii for winning the Little League World Series! You make America very proud.",
        ...
    ]
}
    </code>
</pre>

Over in the terminal running our `server.js` process, you should see logs from the `curl` query. If something went wrong, it will likely appear here.

<pre class="code">
    <code class="plain" data-wrap="false">
[server] Listening on http://0.0.0.0:3000
[twitter] Got 200 new tweets. Total 200
[twitter] Got 200 new tweets. Total 400
[twitter] Got 200 new tweets. Total 600
[twitter] Got 200 new tweets. Total 800
[twitter] Got 200 new tweets. Total 1000
[twitter] Got 200 new tweets. Total 1200
[twitter] Got 200 new tweets. Total 1400
[twitter] Got 200 new tweets. Total 1600
[twitter] Got 200 new tweets. Total 1800
[twitter] Got 200 new tweets. Total 2000
[twitter] Got 200 new tweets. Total 2200
[twitter] Got 200 new tweets. Total 2400
[twitter] Got 199 new tweets. Total 2599
[twitter] Got 199 new tweets. Total 2798
[twitter] Got 198 new tweets. Total 2996
[twitter] Got 200 new tweets. Total 3196
[twitter] Got 36 new tweets. Total 3232
[twitter] Got 1 new tweets. Total 3233
[server] /api/:user got tweets for user barackobama
    </code>
</pre>

We'll use this server to download tweets later in the tutorial, so leave it running. In the meantime, let's talk a bit about *transfer learning*.

## Transfer Learning

Transfer learning is the process of using knowledge gained from one task to solve another task. In practice, this technique involves re-using model weights that were pre-trained using a large dataset as the initial weights of a new model trained using a smaller dataset. In non-transfer learning scenarios model weights are initialized using a random distribution. With transfer learning, a new model's weights are initialized using a checkpoint from a model that was trained using a different dataset, loss function, and/or performance metric.

The intuition behind transfer learning is that knowledge gained from one task can be transferred, through shared model weights, to a different but related task. In a character-level text generation task, our model must learn to extract language patterns entirely from scratch using the training data. Our untrained RNN model has no conception of the english language. Before it can learn to string related words together to form realistic looking sentences, it must learn to combine the right characters to create words at all. If the training data is too small, it's likely that our model won't even be able to generate english looking text, let alone anything that looks like a tweet.

Twitter's API restricts tweet downloads to a mere [3,200 tweets per user account](https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline.html), which isn't much data at all. If we were to train a model with randomly initialized weights using only tweets from a single user's account as training data, the model would perform very poorly. I would expect the model to either not be able to extract useful language patterns from such little text, or to instead memorize the training data and output only exact samples found in the training set. Instead of training our individual twitter user models using randomly initialized weights, we will instead initialize them using the weights of our base model, which we trained using over seven million tweets in [Part 3](twitterbot-part-3-model-inference-and-deployment.html). Our base model has already learned to create english words and sentences, the appropriate lengths of tweets, and how to RT, @mention, and #hashtag. This prior knowledge extends our capability to train a model to imitate an individual twitter user using very little training data.

Here's a pseudo-code example of how the model fine-tuning process works using transfer learning.

<pre class="code">
    <code class="javascript" data-wrap="false">
// train a new model for a very long time on a very large dataset
const baseModel = train(createModel(), 'large-dataset.txt')

// fine-tune the pre-trained model using a small dataset
const fineTunedModel = train(baseModel, 'small-dataset.txt')
    </code>
</pre>

## Twitter Application