import pandas as pd
import numpy as np
from utils.utils import create_topic_columns
from IPython.display import display
from utils.plotting import plot_channel_stats, plot_compressed_channel_stats, plot_sentiment_series
from sentiments import getPolarity
import seaborn as sns
import matplotlib.pyplot as plt

cnnDataset=pd.read_csv("E:/Python/Python/SIN/Twitter/Dataset/CNNnews18_10000.csv")
indiaTodayDataset=pd.read_csv("E:/Python/Python/SIN/Twitter/Dataset/IndiaToday_10000.csv")
ndtvDataset=pd.read_csv("E:/Python/Python/SIN/Twitter/Dataset/ndtv_10000.csv")
timesNowDataset=pd.read_csv("E:/Python/Python/SIN/Twitter/Dataset/TimesNow_10000.csv")

tweets=pd.concat([cnnDataset,indiaTodayDataset,ndtvDataset,timesNowDataset])



topics = pd.read_csv('E:/Python/Python/SIN/Twitter/utilsDataset/topics.csv')
topics[['title', 'slug']]

channels = pd.read_csv('E:/Python/Python/SIN/Twitter/utilsDataset/channels.csv')
channels[['title','color']]


tweets=create_topic_columns(tweets,topics)


num_relevant = tweets.relevant.sum()
num_total =tweets.shape[0]
print ('Number of relevant tweets: %s' % num_relevant)
print ('Total number of tweets: %s' % num_total)
print ('Percentage of relevant tweets: %0.2f%%' % (100*num_relevant/num_total))

channel_stats = pd.DataFrame({
    'relevant': tweets.groupby('User').relevant.sum().astype(int),
    'total': tweets.groupby('User').size()                           
})
channel_stats['percentage_relevant'] = (100*channel_stats.relevant/channel_stats.total).round(2)
channel_stats.sort_values('percentage_relevant', ascending=False)

tweets=tweets[tweets.relevant]

absolutes = tweets.groupby('User')[topics.slug].sum().astype(int)

print("Number of tweets related to each topic by each channel:\n")
display(absolutes)

totals = tweets.groupby('User').size()
relatives = 100 * absolutes.divide(totals, axis=0)

print("percentage of tweets related to each topic by each channel:\n")
display(relatives)

plot_channel_stats(relatives, topics, channels, title='Relative topic coverage\n(% of total # of each channel\'s tweets)')

tweets['sentiment_score']=tweets['Tweet'].apply(getPolarity)


sns.histplot(data=tweets, x="sentiment_score", element="step", color="black")
plt.xlabel("Sentiment Score")
plt.title('Sentiment scores distribution')
plt.gca().get_yaxis().set_visible(False)
plt.xlim(-1,1)
plt.show()


scores = pd.DataFrame(index=channels.sort_values('title').title, columns=topics.slug, )
for User, group in tweets.groupby('User'):
    for topic in topics.slug:
        scores.loc[User, topic] = group[group[topic]].sentiment_score.mean()
scores = scores.rename_axis('Topic', axis=1)
scores = scores.rename_axis('User', axis=0)

print("Average sentimental scores related to each topic by each channel:\n")
scores=scores.iloc[4:,:].fillna(0)
print(scores)




plot_channel_stats(scores, topics, channels, fig_height=10, y_center=True, title='Average sentiment by topic')



plot_compressed_channel_stats(scores, y_center=True, title='Average sentiment by topic')



# plot_sentiment_series(tweets, topics, channels, start_date=datetime(2015, 1, 1), title='Sentiment evolution according to timeline')









