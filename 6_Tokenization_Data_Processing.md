# Module 6: Tokenization & Data Processing

### Tokenization Strategies: BPE, WordPiece, Unigram
- **Description**: Learn various tokenization methods to convert text into model-readable tokens.
- **Concepts Covered**: `tokenization`, `BPE`, `WordPiece`, `Unigram`
- **Learning Resources**:
  - [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)
  - [Byte Pair Encoding (BPE) Explained](https://leimao.github.io/blog/Byte-Pair-Encoding/)
- **Tools**:
  - [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)
  - [SentencePiece](https://github.com/google/sentencepiece)

### Custom Tokenizer Training
- **Description**: Train custom tokenizers tailored to specific datasets or domains.
- **Concepts Covered**: `custom tokenizers`, `tokenizer training`, `domain-specific`, `vocabulary optimization`
- **Learning Resources**:
  - [Hugging Face: Training a Tokenizer](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html)
  - [SentencePiece Training Guide](https://github.com/google/sentencepiece#train-sentencepiece-model)
  - [SmolGPT Implementation](https://github.com/Om-Alve/smolGPT) - Real-world example of domain-specific tokenizer training
  - [GPT-2 Implementation from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Karpathy's comprehensive guide
  - [llama2.c Repository](https://github.com/karpathy/llama2.c) - Reference implementation
- **Tools**:
  - [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/index)
  - [SentencePiece](https://github.com/google/sentencepiece)
- **Key Insights**:
  - Domain-specific tokenizers can be 8x more efficient (e.g., 4K vocab vs 32-50K)
  - Smaller vocabulary reduces embedding layer size and training compute
  - Essential for resource-constrained training scenarios

### Data Collection
- **Description**: Implement large-scale data collection strategies for training LLMs.
- **Concepts Covered**: `web crawling`, `distributed scraping`, `data archival`, `stream processing`, `social media scraping`
- **Learning Resources**:
  - [Common Crawl Documentation](https://commoncrawl.org/the-data/get-started/)
  - [Distributed Web Scraping Guide](https://www.scrapingbee.com/blog/distributed-web-scraping/)
  - [Best Scraping Tools Directory](https://bestscrapingtools.com/web-crawling-tools/)
- **Tools**:
  - Web Archives:
    - [Common Crawl](https://commoncrawl.org/) - Petabyte-scale web archive
    - [Internet Archive](https://archive.org/web/) - Historical web snapshots
  - Distributed Crawlers:
    - [Scrapy](https://scrapy.org/) with [ScrapyD](https://scrapyd.readthedocs.io/)
    - [Colly](https://github.com/gocolly/colly) - Go-based crawler
    - [Spider-rs](https://github.com/spider-rs/spider) - Fast open-source crawler with Python bindings
    - [InstantAPI.ai](https://web.instantapi.ai) - AI-powered web scraping solution
  - Stream Processing:
    - [Apache Kafka](https://kafka.apache.org/) - Real-time data pipelines
    - [Apache Spark](https://spark.apache.org/) - Large-scale data processing
  - **Key Features to Consider**:
    - Lazy loading support
    - Social media data extraction
    - Rate limiting and compliance
    - Dynamic content handling
    - Cross-platform compatibility

### Data Cleaning & Preprocessing Pipelines
- **Description**: Build robust pipelines to clean, normalize, and prepare text data for LLMs.
- **Concepts Covered**: `data cleaning`, `preprocessing`, `normalization`, `pipeline`
- **Learning Resources**:
  - [Data Cleaning with Python](https://www.kaggle.com/learn/data-cleaning)
  - [Text Preprocessing Techniques](https://towardsdatascience.com/8-steps-to-master-data-preparation-with-python-85555d45f54b)
- **Tools**:
  - [spaCy](https://spacy.io/)
  - [NLTK](https://www.nltk.org/)

### Pre-training Datasets
- **Description**: Explore and utilize large-scale datasets suitable for pre-training language models, focusing on diverse, high-quality text corpora.
- **Concepts Covered**: `web crawling`, `data curation`, `quality filtering`, `deduplication`, `content diversity`, `multilingual data`, `domain-specific corpora`
- **Learning Resources**:
  - [RedPajama Data Processing Guide](https://github.com/togethercomputer/RedPajama-Data)
  - [Building High-Quality Pre-training Corpora](https://arxiv.org/abs/2010.12741)
  - [The Pile: An 800GB Dataset of Diverse Text](https://pile.eleuther.ai/)
  - [SlimPajama Technical Report](https://arxiv.org/abs/2401.07608)
- **Tools**:
  - Dataset Creation:
    - [Datasets-CLI](https://github.com/huggingface/datasets-cli) - Command-line tool for dataset management
    - [FastText Language Detection](https://fasttext.cc/docs/en/language-identification.html)
    - [Deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)
    - [CCNet Processing Tools](https://github.com/facebookresearch/cc_net)
- **Popular Datasets**:
  - General Purpose:
    - [The Pile](https://pile.eleuther.ai/) - 825GB diverse English text
    - [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) - 1.2T tokens
    - [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) - 627B tokens, cleaned version
    - [C4](https://huggingface.co/datasets/c4) - Colossal Clean Crawled Corpus
    - [ROOTS](https://huggingface.co/datasets/bigscience-data/roots) - Multilingual dataset
  - Domain-Specific:
    - [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/) - Biomedical literature
    - [ArXiv Dataset](https://huggingface.co/datasets/arxiv_dataset) - Scientific papers
    - [GitHub Code](https://huggingface.co/datasets/codeparrot/github-code) - Programming code
  - Multilingual:
    - [mC4](https://huggingface.co/datasets/mc4) - Multilingual C4
    - [OSCAR](https://huggingface.co/datasets/oscar) - Open Super-large Crawled Aggregated coRpus