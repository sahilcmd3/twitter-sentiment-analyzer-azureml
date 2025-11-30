# Twitter Sentiment Analysis Bot

A comprehensive sentiment analysis system for Twitter data, powered by Azure Machine Learning and BERT (DistilBERT) deep learning models.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![Azure](https://img.shields.io/badge/Azure-ML-0078D4)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **Real-time Twitter Sentiment Analysis** - Analyze tweets for any search query
- **BERT-based Deep Learning** - Fine-tuned DistilBERT model with 99%+ accuracy
- **Modern Web Interface** - Dark theme UI with neon accents
- **Azure Cloud Deployment** - Scalable, production-ready architecture
- **Multiple ML Models** - BERT, Logistic Regression, SVM, Naive Bayes
- **Interactive Visualizations** - Sentiment distribution charts and timelines

## 🏗️ Architecture

```
User (Browser)
    ↓
Azure App Service (Flask Web App)
    ↓
Azure ML Endpoint (BERT Model)
    ↓
Azure Blob Storage (Model Files)
```

## 🚀 Live Demo

Access the deployed application:
```
https://sentiment-bot-48a4f2d5.azurewebsites.net
```

## 📋 Prerequisites

- Python 3.11+
- Azure Account (free tier available)
- Twitter API Bearer Token
- Git

## 🔧 Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/az-sentiment.git
cd az-sentiment
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# or
.venv\Scripts\activate         # Windows CMD
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file:
```env
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
AZURE_ML_ENDPOINT=your_azure_ml_endpoint_url
AZURE_ML_API_KEY=your_azure_ml_api_key
```

5. **Download pre-trained models**

Models are hosted on Azure Blob Storage. Configure URLs in `download_models.py`:
```python
MODEL_URLS = {
    'bert_sentiment.zip': 'https://your-storage.blob.core.windows.net/models/bert_sentiment.zip',
    'best_model.pkl': 'https://your-storage.blob.core.windows.net/models/best_model.pkl',
    'vectorizer.pkl': 'https://your-storage.blob.core.windows.net/models/vectorizer.pkl',
}
```

Then run:
```bash
python download_models.py
```

6. **Run locally**
```bash
python main.py
```

Access at: `http://localhost:5000`

## ☁️ Azure Deployment

### Deploy Azure ML Endpoint

```bash
# Update subscription ID in deploy_to_azureml.py
python deploy_to_azureml.py
```

### Deploy Web App

Follow the comprehensive guide in `AZURE_APP_SERVICE_DEPLOYMENT.md`

Or use the automated script:
```bash
python configure_webapp.py  # Configure settings
# Then deploy via VS Code or Azure Portal
```

## 📊 Training Models

Train BERT model on Sentiment140 dataset (1.6M tweets):

```bash
python scripts/train_bert.py
```

Train traditional ML models:
```bash
python scripts/train_models.py
```

## 🧪 Testing

Test Azure ML endpoint:
```bash
python test_deployment.py
```

Test with curl:
```bash
curl -X POST https://your-endpoint.azurewebsites.net/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "python programming", "max_tweets": 50}'
```

## 📁 Project Structure

```
az-sentiment/
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── sentiment_analyzer.py   # Multi-model sentiment analysis
│   │   ├── text_preprocessing.py    # NLP preprocessing
│   │   ├── twitter_collector.py     # Twitter API integration
│   │   └── visualizer.py            # Data visualization
│   ├── routes/
│   │   ├── api_routes.py            # REST API endpoints
│   │   └── main_routes.py           # Web interface routes
│   ├── templates/
│   │   └── index.html               # Frontend UI
│   └── models/                      # Pre-trained models (excluded from Git)
├── scripts/
│   ├── train_bert.py                # BERT training script
│   └── train_models.py              # Traditional ML training
├── data/
│   └── sentiment140.csv             # Training dataset (excluded from Git)
├── main.py                          # Application entry point
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment for Azure ML
├── score.py                         # Azure ML inference script
├── deploy_to_azureml.py            # Azure ML deployment automation
├── configure_webapp.py              # Azure App Service configuration
├── download_models.py               # Model download utility
└── README.md
```

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TWITTER_BEARER_TOKEN` | Twitter API v2 bearer token | Yes (for Twitter data) |
| `AZURE_ML_ENDPOINT` | Azure ML endpoint URL | Yes (for cloud inference) |
| `AZURE_ML_API_KEY` | Azure ML endpoint key | Yes (for cloud inference) |

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| BERT (DistilBERT) | 99.6% | 0.996 | 0.996 | 0.996 |
| Logistic Regression | 78.2% | 0.782 | 0.782 | 0.782 |
| SVM | 77.8% | 0.778 | 0.778 | 0.778 |
| Naive Bayes | 75.5% | 0.755 | 0.755 | 0.755 |

*Trained on Sentiment140 dataset (1.6M tweets)*

## 💰 Azure Costs

| Resource | Tier | Monthly Cost |
|----------|------|--------------|
| App Service | B1 Basic | ~$13 |
| App Service | F1 Free | $0 (limited) |
| Azure ML Endpoint | Standard_DS2_v2 | ~$125 |
| Blob Storage | Standard | ~$0.10 |
| **Total** | | **~$138/month** |

*Scale down ML endpoint when not in use to reduce costs*

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Sahil & Aryan** - *Initial work*

## 🙏 Acknowledgments

- Sentiment140 dataset by Stanford University
- Hugging Face Transformers library
- Azure Machine Learning team
- Twitter API v2

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check deployment guides in the `docs/` folder
- Review Azure ML logs for debugging

## 🗺️ Roadmap

- [ ] Add Cosmos DB for analysis history
- [ ] Implement real-time streaming analysis
- [ ] Add support for multiple languages
- [ ] Create mobile app interface
- [ ] Add user authentication
- [ ] Implement rate limiting and caching

---

**Built with ❤️ for research in sentiment analysis**
