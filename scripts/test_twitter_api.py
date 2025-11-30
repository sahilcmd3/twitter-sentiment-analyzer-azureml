import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.twitter_collector import TwitterCollector
from dotenv import load_dotenv
import json

load_dotenv()


def main():
    print(
        """
    ╔══════════════════════════════════════════════════════════╗
    ║   Twitter API Connection Test                            ║
    ║   Author: sahil & aryan                                  ║
    ╚══════════════════════════════════════════════════════════╝
    """
    )

    # Initialize collector
    print("Initializing Twitter Collector...")
    collector = TwitterCollector()

    # Check status
    status = collector.get_api_status()

    print("\nAPI Status:")
    print(f"Bearer Token Configured: {'good' if status['bearer_token_configured'] else 'bad'}")
    print(f"Client Initialized: {'good' if status['client_initialized'] else 'bad'}")
    print(f"Using Real API: {'good' if status['using_real_api'] else 'bad (Mock Data)'}")

    if not status["using_real_api"]:
        print("\nNot using real Twitter API")
        print("To use real API:")
        print("1. Get credentials from https://developer.twitter.com/")
        print("2. Update .env file with TWITTER_BEARER_TOKEN")
        print("3. Run this script again")
        print("\nFor now, testing with mock data...\n")

    # Test queries
    test_queries = ["Python programming", "Machine Learning", "Cloud Computing"]

    print("\n" + "=" * 60)
    print("Testing with sample queries")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        try:
            tweets = collector.search_tweets(query, max_results=5)

            print(f"Retrieved {len(tweets)} tweets")

            if tweets:
                print("\nSample tweet:")
                tweet = tweets[0]
                print(f"Author: @{tweet['author_username']}")
                print(f"Text: {tweet['text'][:100]}...")
                print(f"Likes: {tweet['likes']} | Retweets: {tweet['retweets']}")
                print(f"Created: {tweet['created_at']}")

        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Twitter API test complete!")
    print("=" * 60)

    if status["using_real_api"]:
        print("\nYou're ready to analyze real tweets!")
    else:
        print("\nConfigure Twitter API to analyze real tweets")

    print("\nNext steps:")
    print("1. Run the web app: python main.py")
    print("2. Train ML models: python scripts/train_models.py")
    print("3. Test models: python scripts/test_models.py")


if __name__ == "__main__":
    main()
