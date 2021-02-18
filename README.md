# Comments Analyzer
Comments-Analyzer is a website aimed at analyzing comments made on reddit posts. What happens is the user provides the website with a post URL, then Comments-Analyzer then scrapes
the comments from the aformentioned post and then return with an analysis of of all the comments within that particuler post. 

**CURRENT VERSION: 2.1**

## V 1.0
**1.1**:  There are two models at the moment: The general model and the news-tailored model. Now only the general model has been implemented but the second model has already been built. So the plan here is to give the option to use either model to predict.

**1.2**: I have added a third model called "Football". There is a simple HTML dropdown list where the user can choose which model to predict with depending on the type of subreddit post it is. 

**1.3**: I started on the front-end section, there is a navbar and the textbox and button look ugly instead of super ugly. Next I wish to finish it up in order to move on to the next social media platform.

## V 2.0
**2.1**: I have started and finished the second phase using twitter API. Now the application can fetch tweets from hashtags and/or replied to tweets. Currently I am using the free version of the twitter API so of course there is limitation to the amount of information retrieved.  

**2.2**: I am done with the general model and so far workds fine. Now I'm in the process of a new approach: Created two classification models for predicting football tweets and using them both to predict and comapre their confidence levels to pick the best prediction. As of this moment this has yet to be implemented.
