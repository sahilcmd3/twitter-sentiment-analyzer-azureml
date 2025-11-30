import logging
import os
from dotenv import load_dotenv

from flask import Flask
from flask_cors import CORS

load_dotenv()


def create_app():
    app = Flask(__name__)

    # Config
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key")
    app.config["MAX_TWEETS"] = int(os.getenv("MAX_TWEETS_PER_REQUEST", 100))

    # Enable CORS
    CORS(app)

    # Logging
    logging.basicConfig(level=logging.INFO)

    # Register routes
    from app.routes import api_bp, main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
