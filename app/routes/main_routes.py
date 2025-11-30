from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Home page"""
    return render_template("index.html")


@main_bp.route("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Twitter Sentiment Analysis Bot",
        "author": "sahil & aryan",
        "version": "1.0.0",
    }, 200


@main_bp.route("/about")
def about():
    """About page"""
    return {
        "project": "Twitter Sentiment Analysis",
        "author": "sahil & aryan",
        "description": "Cloud-based real-time sentiment analysis using Azure",
    }, 200
