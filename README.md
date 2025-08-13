# Infact 📰

An open-source implementation of a Ground News-like news aggregation service that desensationalizes public news through AI-powered clustering and fact extraction.

## Overview

Infact demonstrates an intelligent news processing pipeline that takes sensationalized articles and transforms them into factual, neutral reporting. The system clusters similar stories, extracts key facts, removes editorial bias, and generates clean, informative articles.

## 🚀 Key Features

- **Smart Article Clustering**: Groups related news stories using advanced NLP embeddings
- **Fact vs Opinion Separation**: Automatically distinguishes between factual content and editorial musings
- **Bias Reduction**: Removes sensationalized language while preserving core information
- **Duplicate Detection**: Merges similar facts to eliminate redundancy
- **Topic Modeling**: Automatically names story clusters using LDA topic extraction
- **Interactive Demo**: Real-time processing pipeline with visual analytics

## 🛠️ Tech Stack

### Core Processing Pipeline
- **NLP Engine**: spaCy for text preprocessing and named entity recognition
- **Embeddings**: Sentence Transformers (all-mpnet-base-v2) for semantic similarity
- **Clustering**: Scikit-learn KMeans with TF-IDF feature enhancement
- **Topic Modeling**: Gensim LDA for automatic cluster naming
- **Sentiment Analysis**: TextBlob for opinion detection
- **Similarity Matching**: FuzzyWuzzy for duplicate detection

### AI & Generation
- **LLM Integration**: Google Gemini 2.5 Flash for article generation
- **Orchestration**: LangChain for prompt management
- **GPU Acceleration**: CUDA support for faster embeddings

### Visualization & Interface
- **Frontend**: Streamlit for interactive demo
- **Charts**: Plotly for cluster visualization and analytics
- **Word Clouds**: Visual topic representation
- **Real-time Processing**: Live pipeline monitoring

## 🏗️ Architecture

The pipeline processes news articles through six distinct stages:

1. **Preprocessing**: Text cleaning, tokenization, and normalization
2. **Clustering**: Semantic grouping of related articles
3. **Topic Extraction**: Automatic naming using LDA topic modeling
4. **Fact Extraction**: Separation of facts from opinions using NER and sentiment analysis
5. **Deduplication**: Merging similar facts to reduce redundancy
6. **Article Generation**: Creating neutral, factual articles using LLM

## 📊 Efficiency & Scalability

### Performance Optimizations
- **Batch Processing**: Embeddings generated in configurable batches (default: 32)
- **Memory Management**: Text truncation for large articles (1M char limit)
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Caching**: Model loading cached to prevent reinitialization
- **Parallel Processing**: Independent operations run simultaneously

### Scalability Features
- **Modular Design**: Each pipeline stage is independently scalable
- **Configurable Clustering**: Dynamic cluster count based on article volume
- **Resource Limits**: Built-in safeguards for memory and processing time
- **Streaming Ready**: Architecture supports real-time article ingestion

## 🎯 Demo Data

The project includes extremely clickbait sample articles to test the desensationalization process:

```json
[
  {
    "title": "Tech Company Announces SHOCKING AI Initiative That Will DESTROY Competition!",
    "content": "Industry experts are STUNNED by this revolutionary announcement..."
  }
]
```

These exaggerated examples demonstrate the system's ability to extract factual content from sensationalized reporting.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google API key for Gemini
- NGROK token (for Colab deployment)

### Installation

```bash
pip install streamlit sentence-transformers scikit-learn gensim spacy textblob fuzzywuzzy python-Levenshtein
pip install google-generativeai langchain langchain-google-genai pyngrok wordcloud plotly nltk

# Download required models
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
```

### Running the Demo

1. Set up environment variables:
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export NGROK_AUTH_TOKEN="your_ngrok_token"  # For Colab only
```

2. Run the Jupyter notebook:
```bash
jupyter notebook Infact.ipynb
```

3. Or extract and run the Streamlit app:
```bash
streamlit run app.py
```

## 📈 Pipeline Metrics

The system provides comprehensive analytics:

- **Processing Speed**: < 1 minute for 20 articles
- **Cluster Accuracy**: Visualized through PCA projections
- **Fact Extraction Rate**: Typically 5-10 facts per article
- **Deduplication Efficiency**: 20-40% reduction in redundant content
- **Bias Reduction**: Measured through sentiment analysis

## 🔧 Configuration

### Clustering Parameters
```python
n_clusters = min(max(3, len(texts) // 20), 15)  # Dynamic cluster sizing
threshold = 0.7  # Similarity threshold for deduplication
batch_size = 32  # Embedding batch size
```

### Model Settings
```python
sentence_model = 'all-mpnet-base-v2'  # Embedding model
llm_model = 'gemini-2.5-flash'        # Generation model
max_text_length = 1000000             # Processing limit
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- Additional news source integrations
- Enhanced bias detection algorithms
- Real-time processing capabilities
- Multi-language support
- Advanced fact-checking integration

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Use Cases

- **News Organizations**: Reduce editorial bias in reporting
- **Research**: Study media bias and sensationalization patterns
- **Education**: Demonstrate AI applications in journalism
- **Fact-Checking**: Automated extraction of verifiable claims
- **Content Moderation**: Identify opinion vs factual content

---

*Infact: Making news more factual, one article at a time.*
