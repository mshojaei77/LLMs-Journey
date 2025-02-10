# Module 6: Tokenization & Data Processing

### Tokenization Strategies: BPE, WordPiece, Unigram
- **Description**: Learn various tokenization methods to convert text into model-readable tokens.
- **Concepts Covered**: `tokenization`, `BPE`, `WordPiece`, `Unigram`
- **Learning Resources**:
  - [![Hugging Face Tokenizers](https://badgen.net/badge/Docs/Hugging%20Face%20Tokenizers/green)](https://huggingface.co/docs/tokenizers/index)
  - [![Byte Pair Encoding (BPE) Explained](https://badgen.net/badge/Blog/Byte%20Pair%20Encoding%20Explained/cyan)](https://leimao.github.io/blog/Byte-Pair-Encoding/)
- **Tools**:
  - [![Hugging Face Tokenizers](https://badgen.net/badge/Framework/Hugging%20Face%20Tokenizers/green)](https://huggingface.co/docs/tokenizers/index)
  - [![SentencePiece](https://badgen.net/badge/Github%20Repository/SentencePiece/gray)](https://github.com/google/sentencepiece)

### Custom Tokenizer Training
- **Description**: Train custom tokenizers tailored to specific datasets or domains.
- **Concepts Covered**: `custom tokenizers`, `tokenizer training`, `domain-specific`, `vocabulary optimization`
- **Learning Resources**:
  - [![Hugging Face: Training a Tokenizer](https://badgen.net/badge/Docs/Training%20a%20Tokenizer/green)](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html)
  - [![SentencePiece Training Guide](https://badgen.net/badge/Docs/SentencePiece%20Training%20Guide/green)](https://github.com/google/sentencepiece#train-sentencepiece-model)
  - [![SmolGPT Implementation](https://badgen.net/badge/Github%20Repository/SmolGPT/gray)](https://github.com/Om-Alve/smolGPT)
  - [![GPT-2 Implementation from Scratch](https://badgen.net/badge/Video/GPT-2%20Implementation%20from%20Scratch/red)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
  - [![llama2.c Repository](https://badgen.net/badge/Github%20Repository/llama2.c/gray)](https://github.com/karpathy/llama2.c)
- **Tools**:
  - [![Hugging Face Tokenizers](https://badgen.net/badge/Framework/Hugging%20Face%20Tokenizers/green)](https://huggingface.co/docs/tokenizers/index)
  - [![SentencePiece](https://badgen.net/badge/Github%20Repository/SentencePiece/gray)](https://github.com/google/sentencepiece)
- **Key Insights**:
  - Domain-specific tokenizers can be 8x more efficient (e.g., 4K vocab vs 32-50K)
  - Smaller vocabulary reduces embedding layer size and training compute
  - Essential for resource-constrained training scenarios

### Data Collection
- **Description**: Implement large-scale data collection strategies for training LLMs.
- **Concepts Covered**: `web crawling`, `distributed scraping`, `data archival`, `stream processing`, `social media scraping`
- **Learning Resources**:
  - [![Common Crawl Documentation](https://badgen.net/badge/Docs/Common%20Crawl%20Documentation/green)](https://commoncrawl.org/the-data/get-started/)
  - [![Distributed Web Scraping Guide](https://badgen.net/badge/Blog/Distributed%20Web%20Scraping%20Guide/cyan)](https://www.scrapingbee.com/blog/distributed-web-scraping/)
  - [![Best Scraping Tools Directory](https://badgen.net/badge/Website/Best%20Scraping%20Tools%20Directory/blue)](https://bestscrapingtools.com/web-crawling-tools/)
- **Tools**:
  - Web Archives:
    - [![Common Crawl](https://badgen.net/badge/Website/Common%20Crawl/blue)](https://commoncrawl.org/)
    - [![Internet Archive](https://badgen.net/badge/Website/Internet%20Archive/blue)](https://archive.org/web/)
  - Distributed Crawlers:
    - [![Scrapy](https://badgen.net/badge/Framework/Scrapy/green)](https://scrapy.org/)
    - [![Colly](https://badgen.net/badge/Github%20Repository/Colly/gray)](https://github.com/gocolly/colly)
    - [![Spider-rs](https://badgen.net/badge/Github%20Repository/Spider-rs/gray)](https://github.com/spider-rs/spider)
    - [![InstantAPI.ai](https://badgen.net/badge/API%20Provider/InstantAPI.ai/blue)](https://web.instantapi.ai)
  - Stream Processing:
    - [![Apache Kafka](https://badgen.net/badge/Framework/Apache%20Kafka/green)](https://kafka.apache.org/)
    - [![Apache Spark](https://badgen.net/badge/Framework/Apache%20Spark/green)](https://spark.apache.org/)
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
  - [![Data Cleaning with Python](https://badgen.net/badge/Tutorial/Data%20Cleaning%20with%20Python/blue)](https://www.kaggle.com/learn/data-cleaning)
  - [![Text Preprocessing Techniques](https://badgen.net/badge/Blog/Text%20Preprocessing%20Techniques/cyan)](https://towardsdatascience.com/8-steps-to-master-data-preparation-with-python-85555d45f54b)
- **Tools**:
  - [![spaCy](https://badgen.net/badge/Framework/spaCy/green)](https://spacy.io/)
  - [![NLTK](https://badgen.net/badge/Framework/NLTK/green)](https://www.nltk.org/)

### Pre-training Datasets
- **Description**: Explore and utilize large-scale datasets suitable for pre-training language models, focusing on diverse, high-quality text corpora.
- **Concepts Covered**: `web crawling`, `data curation`, `quality filtering`, `deduplication`, `content diversity`, `multilingual data`, `domain-specific corpora`
- **Learning Resources**:
  - [![RedPajama Data Processing Guide](https://badgen.net/badge/Github%20Repository/RedPajama%20Data%20Processing%20Guide/gray)](https://github.com/togethercomputer/RedPajama-Data)
  - [![Building High-Quality Pre-training Corpora](https://badgen.net/badge/Paper/Building%20High-Quality%20Pre-training%20Corpora/purple)](https://arxiv.org/abs/2010.12741)
  - [![The Pile: An 800GB Dataset of Diverse Text](https://badgen.net/badge/Website/The%20Pile/blue)](https://pile.eleuther.ai/)
  - [![SlimPajama Technical Report](https://badgen.net/badge/Paper/SlimPajama%20Technical%20Report/purple)](https://arxiv.org/abs/2401.07608)
- **Tools**:
  - Dataset Creation:
    - [![Datasets-CLI](https://badgen.net/badge/Github%20Repository/Datasets-CLI/gray)](https://github.com/huggingface/datasets-cli)
    - [![FastText Language Detection](https://badgen.net/badge/Framework/FastText%20Language%20Detection/green)](https://fasttext.cc/docs/en/language-identification.html)
    - [![Deduplicate-text-datasets](https://badgen.net/badge/Github%20Repository/Deduplicate-text-datasets/gray)](https://github.com/google-research/deduplicate-text-datasets)
    - [![CCNet Processing Tools](https://badgen.net/badge/Github%20Repository/CCNet%20Processing%20Tools/gray)](https://github.com/facebookresearch/cc_net)
- **Popular Datasets**:
  - General Purpose:
    - [![The Pile](https://badgen.net/badge/Hugging%20Face%20Dataset/The%20Pile/yellow)](https://pile.eleuther.ai/)
    - [![RedPajama](https://badgen.net/badge/Hugging%20Face%20Dataset/RedPajama/yellow)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)
    - [![SlimPajama](https://badgen.net/badge/Hugging%20Face%20Dataset/SlimPajama/yellow)](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
    - [![C4](https://badgen.net/badge/Hugging%20Face%20Dataset/C4/yellow)](https://huggingface.co/datasets/c4)
    - [![ROOTS](https://badgen.net/badge/Hugging%20Face%20Dataset/ROOTS/yellow)](https://huggingface.co/datasets/bigscience-data/roots)
  - Domain-Specific:
    - [![PubMed Central](https://badgen.net/badge/Website/PubMed%20Central/blue)](https://www.ncbi.nlm.nih.gov/pmc/)
    - [![ArXiv Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/ArXiv%20Dataset/yellow)](https://huggingface.co/datasets/arxiv_dataset)
    - [![GitHub Code](https://badgen.net/badge/Hugging%20Face%20Dataset/GitHub%20Code/yellow)](https://huggingface.co/datasets/codeparrot/github-code)
  - Multilingual:
    - [![mC4](https://badgen.net/badge/Hugging%20Face%20Dataset/mC4/yellow)](https://huggingface.co/datasets/mc4)
    - [![OSCAR](https://badgen.net/badge/Hugging%20Face%20Dataset/OSCAR/yellow)](https://huggingface.co/datasets/oscar)