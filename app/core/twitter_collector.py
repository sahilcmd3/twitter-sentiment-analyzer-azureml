import os
import tweepy
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TwitterCollector:
    def __init__(self):
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret = os.getenv("TWITTER_API_SECRET")
        self.client = None
        self.using_real_api = False

        self._initialize_client()

    def _initialize_client(self):
        """Initialize Twitter API client"""
        if self.bearer_token and self.bearer_token != "your_bearer_token_here":
            try:
                self.client = tweepy.Client(
                    bearer_token=self.bearer_token, wait_on_rate_limit=True
                )
                # Don't test connection on init to avoid rate limiting
                # Connection will be verified on first actual API call
                self.using_real_api = True
                logger.info("Twitter API client configured (credentials provided)")
            except Exception as e:
                logger.warning(f"Twitter API initialization failed: {e}")
                logger.info("Falling back to mock data")
                self.client = None
        else:
            logger.info("No Twitter API credentials - using mock data")

    def _test_connection(self):
        """Test Twitter API connection"""
        try:
            # Simple test query
            result = self.client.search_recent_tweets(query="test", max_results=10)
            logger.info("Twitter API connection test successful")
        except tweepy.errors.Unauthorized:
            raise Exception("Invalid Twitter API credentials")
        except tweepy.errors.TooManyRequests:
            logger.warning("Rate limit reached")
        except Exception as e:
            raise Exception(f"Twitter API test failed: {str(e)}")

    def search_tweets(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search for tweets

        Args:
            query: Search query string
            max_results: Number of tweets to retrieve (10-100)

        Returns:
            List of tweet dictionaries
        """
        if self.client and self.using_real_api:
            logger.info(f"Using real Twitter API for query: {query}")
            return self._fetch_real_tweets(query, max_results)
        else:
            logger.info(f"Using mock data (API not configured)")
            return self._get_mock_tweets(max_results)

    def _fetch_real_tweets(self, query: str, max_results: int) -> List[Dict]:
        """Fetch real tweets from Twitter API v2"""
        try:
            logger.info(f"Searching Twitter for: {query}")

            # Enhance query to get quality tweets
            enhanced_query = f"{query} -is:retweet lang:en"

            # API call
            tweets = self.client.search_recent_tweets(
                query=enhanced_query,
                max_results=min(max_results, 100),  # API limit
                tweet_fields=["created_at", "public_metrics", "author_id", "lang"],
                expansions=["author_id"],
                user_fields=["username", "name"],
            )

            if not tweets.data:
                logger.warning("No tweets found, using mock data")
                return self._get_mock_tweets(max_results)

            # Extract user info
            users = {}
            if tweets.includes and "users" in tweets.includes:
                users = {user.id: user for user in tweets.includes["users"]}

            # Format results
            result = []
            for tweet in tweets.data:
                user = users.get(tweet.author_id)

                tweet_dict = {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "author_id": tweet.author_id,
                    "author_username": user.username if user else "unknown",
                    "author_name": user.name if user else "Unknown User",
                    "likes": tweet.public_metrics.get("like_count", 0),
                    "retweets": tweet.public_metrics.get("retweet_count", 0),
                    "replies": tweet.public_metrics.get("reply_count", 0),
                    "language": tweet.lang,
                }
                result.append(tweet_dict)

            logger.info(f"Fetched {len(result)} real tweets")
            return result

        except tweepy.errors.TooManyRequests as e:
            logger.error("Rate limit exceeded! Twitter API has daily/monthly limits.")
            logger.error("Please wait 15 minutes or use mock data.")
            logger.error(f"Details: {str(e)}")
            self.using_real_api = (
                False  # Temporarily disable to avoid more rate limit calls
            )
            return self._get_mock_tweets(max_results)

        except tweepy.errors.Unauthorized as e:
            logger.error("Invalid Twitter API credentials.")
            logger.error("Please check your TWITTER_BEARER_TOKEN in .env file")
            logger.error(f"Details: {str(e)}")
            self.using_real_api = False
            return self._get_mock_tweets(max_results)

        except tweepy.errors.Forbidden as e:
            logger.error(
                "Access forbidden. Your API tier may not support this operation."
            )
            logger.error(f"Details: {str(e)}")
            self.using_real_api = False
            return self._get_mock_tweets(max_results)

        except Exception as e:
            logger.error(f"Error fetching tweets: {type(e).__name__}: {e}")
            logger.info("Falling back to mock data")
            return self._get_mock_tweets(max_results)

    def _get_mock_tweets(self, count: int) -> List[Dict]:
        """Generate mock tweets for testing"""
        mock_data = [
            # Positive tweets
            "Python is absolutely amazing for data science! Love working with it! #Python #DataScience",
            "Just deployed my first web app on Azure! This is so exciting! #CloudComputing #Azure",
            "Machine learning is transforming everything! The future is here! #AI #MachineLearning",
            "Successfully trained my first neural network today! Feeling accomplished!",
            "Flask makes web development so easy and intuitive. Great framework! #Flask #WebDev",
            "Natural language processing is absolutely fascinating! #NLP #AI",
            "Loving the Azure Functions for serverless computing! So powerful!",
            "Sentiment analysis project is coming together nicely! #DataScience",
            "The Python community is incredibly helpful and supportive!",
            "Cloud computing has made deployment so much easier! #Azure #Cloud",
            # Negative tweets
            "This bug is driving me absolutely crazy! Been debugging for hours #Programming",
            "I hate when my code doesn't work and I have no idea why",
            "This API documentation is terrible. Makes absolutely no sense! #Frustrated",
            "Why is deployment so complicated? This is so annoying!",
            "Spent the whole day fighting with dependencies. Worst day ever.",
            "This error message is completely useless. No helpful information at all!",
            "Machine learning models are so hard to debug. So frustrating! 😤",
            "Why doesn't this code work? I followed the tutorial exactly! Angry!",
            "This library is poorly documented. Wasting so much time! 😡",
            "Configuration issues are the worst part of development. Hate it!",
            # Neutral tweets
            "Learning about cloud computing and its applications in modern software",
            "Exploring different machine learning algorithms for sentiment analysis",
            "Comparing Azure, AWS, and Google Cloud for my project deployment",
            "Reading about natural language processing techniques and their uses",
            "Investigating the best practices for RESTful API design",
            "Studying the differences between various sentiment analysis approaches",
            "Analyzing the performance metrics of different ML models today",
            "Looking into serverless architecture patterns for scalability",
            "Researching data preprocessing techniques for text analysis",
            "Examining the tradeoffs between different cloud service providers",
        ]

        result = []
        for i in range(min(count, len(mock_data))):
            result.append(
                {
                    "id": f"mock_{i+1}",
                    "text": mock_data[i % len(mock_data)],
                    "created_at": datetime.now(),
                    "author_id": f"user_{(i % 10) + 1}",
                    "author_username": f"user{(i % 10) + 1}",
                    "author_name": f"Mock User {(i % 10) + 1}",
                    "likes": (i * 7) % 100,
                    "retweets": (i * 3) % 50,
                    "replies": (i * 2) % 30,
                    "language": "en",
                }
            )

        logger.info(f"Generated {len(result)} mock tweets")
        return result

    def get_api_status(self) -> Dict:
        """Get API connection status"""
        return {
            "using_real_api": self.using_real_api,
            "bearer_token_configured": bool(
                self.bearer_token and self.bearer_token != "your_bearer_token_here"
            ),
            "client_initialized": self.client is not None,
        }
