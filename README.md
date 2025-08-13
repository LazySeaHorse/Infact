# Infact 📰

An open-source implementation of a Ground News-like news aggregation service that desensationalizes public news through AI-powered clustering and fact extraction.

## Overview

Infact demonstrates an intelligent news processing pipeline that takes sensationalized articles and transforms them into factual, neutral reporting. The system clusters similar stories, extracts key facts, removes editorial bias, and generates clean, informative articles.

> [!NOTE]
> Coded to life with help from Claude 4.1 Opus

## 🚀 Key Features

- **Smart Article Clustering**: Groups related news stories using advanced NLP embeddings
- **Fact vs Opinion Separation**: Automatically distinguishes between factual content and editorial musings
- **Bias Reduction**: Removes sensationalized language while preserving core information
- **Duplicate Detection**: Merges similar facts to eliminate redundancy
- **Topic Modeling**: Automatically names story clusters using LDA topic extraction
- **Interactive Demo**: Real-time processing pipeline with visual analytics

|  |  |
|:--:|:--:|
| ![img1](https://i.postimg.cc/5NhrpJSw/gesrfdg.png) | ![img2](https://i.postimg.cc/4xpSS0Wj/rtyhdt5hg.png) |
| ![img3](https://i.postimg.cc/J4Dgb0Zr/tyjhdtfygh.png) | ![img4](https://i.postimg.cc/HWPRcXJv/ujnftryhjnrdtfyg.png) |


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
  {"title":"You Won’t Believe What Governor Silverstone Is Hiding!","content":"BREAKING: Progressive Beacon Daily has uncovered SHOCKING evidence that Governor Silverstone’s secret offshore accounts teem with illicit payoffs from corporate lobbyists! Documents obtained by our insider reveal hidden transactions totaling MILLIONS funneled through shell companies. Critics say this could spell the end of his political career. If you care about TRANSPARENCY, you NEED to read this exposé before it’s buried forever!"},
  {"title":"Is Silverstone the Most Corrupt Governor Ever?","content":"In a STUNNING revelation, Centrist Times reports Governor Silverstone’s office allegedly processed suspicious wire transfers linked to big-energy giants. Sources claim these funds influenced critical environmental votes. Lawmakers are demanding a full inquiry—could this be the biggest scandal in state history? Our exclusive analysis breaks down every transaction and political ramification. You won’t believe how deep this rabbit hole goes!"},
  {"title":"Expose: Silverstone’s Shady Deals Threaten Our Values","content":"Conservative Watchdog News warns that Governor Silverstone’s dereliction of duty isn’t just immoral—it’s PATRIOTIC BETRAYAL! Leaked financial ledgers allegedly show collaboration with radical green groups aiming to dismantle traditional industries. Experts fear these payoffs will cost thousands of jobs and undermine national security. Lawmakers are mobilizing to strip him of office. Don’t miss this fiery breakdown of treasonous politics!"},
  {"title":"Incredible Breakthrough: Scientists Harness Sunlight Like Never Before!","content":"Progressive Beacon Daily celebrates a MIND-BLOWING invention: researchers at Meridian Institute have developed solar panels that convert 90% of sunlight into energy! This could CRUSH the fossil fuel industry and save the planet. Testing shows devices working under low-light conditions—EVERY home can go green. Environmentalists call it the TECHNOLOGY of the century. Find out how this revolution could slash your bills to zero!"},
  {"title":"Solar Miracle Poised to Rewire Energy Market","content":"Centrist Times reports that a team at Meridian Institute unveiled ultra-efficient solar cells boosting energy conversion rates by 50%. Investors are already lining up to fund mass production. Officials say this could stabilize electricity prices and reduce carbon emissions dramatically. Our experts break down what this means for everyday consumers and the global energy landscape. Could this be the energy shift we’ve all waited for?"},
  {"title":"New Solar Tech Sparks Fears of Industrial Collapse","content":"Conservative Watchdog News ALERT: Meridian Institute’s latest solar innovation threatens to DESTROY American manufacturing! Reports indicate the technology could decimate traditional energy sectors, costing millions of jobs. Critics argue the government will FORCE companies to adopt this UNTESTED system, undermining economic stability. Industry leaders are mobilizing to resist—read our fiery take on how radical science is on track to wreck livelihoods."},
  {"title":"Hollywood’s Biggest Star Files for Divorce—MUST-SEE Details!","content":"Progressive Beacon Daily exposes the intimate details behind A-list actor Jordan Calibre’s shocking divorce filing from indie director Riley West. Sources say Calibre cited “irreconcilable creative differences,” but rumors of infidelity swirl like wildfire! Friends claim West discovered damning text messages. Our exclusive interviews delve into every tearful confrontation and trust-shattering betrayal, plus what it means for Calibre’s upcoming blockbuster release."},
  {"title":"Celebrity Split Shocks Fans Worldwide","content":"Centrist Times reveals actor Jordan Calibre has petitioned for divorce from Riley West after a decade-long marriage. While the pair released a joint statement emphasizing mutual respect, insiders hint at deep artistic disagreements and financial disputes. We break down the timeline of their relationship, the terms of their prenuptial agreement, and what this could mean for their sprawling media empire."},
  {"title":"Star Divorce: Hollywood’s Moral Decay Exposed","content":"Conservative Watchdog News decries Jordan Calibre’s divorce from Riley West as yet another symbol of Hollywood’s crumbling moral fabric. Sources allege West’s radical ideology clashed with Calibre’s family values, prompting this public split. Experts warn this trend undermines societal cohesion. Our explosive report uncovers behind-the-scenes drama, lavish alimony demands, and the culture-war stakes at play in Tinseltown’s latest breakup."},
  {"title":"SKY ALERT: Mysterious Comet Heads Straight for Earth!","content":"Progressive Beacon Daily warns: NASA scientists have detected Comet Talora hurtling toward Earth at BREAKNECK speed! Groundbreaking telescopes estimate a collision chance of 2%. While experts urge calm, conspiracy theorists speculate involvement of secret government satellites. Will we see a celestial spectacle—or total annihilation? Our live updates and expert interviews guide you through every astronomical twist before it’s too late!"},
  {"title":"Comet Talora: Real Risk or Media Circus?","content":"Centrist Times highlights recent NASA data on Comet Talora, currently 70 million km away and tracking a near-Earth trajectory. Officials place the impact probability at less than 1%, forecasting a dazzling sky show rather than disaster. We clarify scientific jargon, weigh expert assessments, and outline safe viewing protocols. Learn what the public should REALLY know amid the swirling cosmic hype."},
  {"title":"Armageddon Incoming? Comet Talora Doom Predictions!","content":"Conservative Watchdog News screams ALERT: Comet Talora might be God’s final judgment on a morally bankrupt world! Prepper communities stockpile supplies as the celestial object grows ominously bright. Though NASA insists there’s “no cause for alarm,” regional pastors call for national prayer days. Could this be the sign we’ve ignored for too long? Discover how this cosmic visitor might expose society’s spiritual failings!"},
  {"title":"Hospital Crisis: ERs Drowning in Patients—MUST READ!","content":"Progressive Beacon Daily uncovers a nationwide EMERGENCY as public hospitals report 200% ER capacity surges amid unprecedented flu and COVID-variant outbreaks. Frontline nurses sound the alarm on staff shortages and dwindling medical supplies. Patients wait HOURS for care. Health advocates demand major funding overhauls to save lives. Our exclusive testimonies reveal heartbreaking stories behind overcrowded wards and the real human cost you won’t believe!"},
  {"title":"ER Overload: What You Need to Know","content":"Centrist Times examines the current strain on emergency departments across the country, attributing it to overlapping flu, COVID-19, and RSV seasons. Hospitals report bed shortages and extended wait times. Officials propose federal grants and rapid staffing incentives to alleviate pressure. We analyze policy options, compare regional responses, and provide practical tips for seeking timely medical attention during the crisis."},
  {"title":"Hospitals on Brink: Government Failures EXPOSED","content":"Conservative Watchdog News BLASTS federal mandates for causing ER meltdowns, with hospitals forced to treat unlawful migrants and non-citizens, leaving locals to suffer. Staff report shutdown threats if they refuse care. Citizens face life-or-death delays while bureaucrats bicker. This is a TAXPAYER SCANDAL! Our fiery investigation names the officials responsible and outlines the radical reforms needed to save American healthcare."},
  {"title":"School’s New AI Pronoun Rules Spark Controversy!","content":"Progressive Beacon Daily reveals TechForward Academy’s radical introduction of AI-driven pronoun enforcement—a digital system that auto-corrects speech for inclusivity! Students are seeing real-time alerts and mandatory sensitivity training. Advocates hail it as the FUTURE of respect; critics decry Orwellian overreach. Our in-depth look explores student reactions, privacy concerns, and the impact on classroom culture you can’t afford to miss."},
  {"title":"AI Pronoun Tool: Balanced Perspectives","content":"Centrist Times reports that TechForward Academy is piloting an AI pronoun-assistant to promote inclusivity. The system flags misgendering and offers immediate guidance. Supporters argue it fosters empathy, while detractors question data security and free speech implications. We present viewpoints from educators, legal experts, and parent groups, plus a side-by-side analysis of the program’s benefits and potential pitfalls."},
  {"title":"School’s AI Pronoun Police: Free Speech Under Siege?","content":"Conservative Watchdog News warns TechForward Academy’s AI pronoun enforcement is the latest step toward THOUGHT CONTROL in schools! Students risk penalties for ‘unintentional’ speech errors, and faculty face dismissal if they push back. This techno-authoritarian nightmare could spread nationwide, stifling dissent and bending education to radical ideology. Read our blistering critique on how AI is weaponized against fundamental liberties!"}
]
```

These exaggerated examples demonstrate the system's ability to extract factual content from sensationalized reporting.

## 🚀 Quick Start

Download the Jupyter notebook (or copy paste content into a new Google Colab notebook cell). Hit run.
If you're using Colab, you'll need to configure GOOGLE_API_KEY and NGROK_AUTH_TOKEN secrets (left panel, key icon).

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
