# Module 6: Data Processing

### Data Collection
#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Common Crawl Documentation](https://badgen.net/badge/Docs/Common%20Crawl%20Documentation/green)](https://commoncrawl.org/the-data/get-started/) | [![Best Scraping Tools Directory](https://badgen.net/badge/Website/Best%20Scraping%20Tools%20Directory/blue)](https://bestscrapingtools.com/web-crawling-tools/) |
| [![Distributed Web Scraping Guide](https://badgen.net/badge/Tutorial/Distributed%20Web%20Scraping%20Guide/blue)](https://www.scrapingbee.com/blog/distributed-web-scraping/) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Common Crawl](https://badgen.net/badge/Framework/Common%20Crawl/green)](https://commoncrawl.org/) | [![Internet Archive](https://badgen.net/badge/Website/Internet%20Archive/blue)](https://archive.org/web/) |
| [![Scrapy](https://badgen.net/badge/Framework/Scrapy/green)](https://scrapy.org/) | [![Colly](https://badgen.net/badge/Github%20Repository/Colly/cyan)](https://github.com/gocolly/colly) |
| [![Apache Kafka](https://badgen.net/badge/Framework/Apache%20Kafka/green)](https://kafka.apache.org/) | [![Spider-rs](https://badgen.net/badge/Github%20Repository/Spider-rs/cyan)](https://github.com/spider-rs/spider) |
| [![Apache Spark](https://badgen.net/badge/Framework/Apache%20Spark/green)](https://spark.apache.org/) | [![InstantAPI.ai](https://badgen.net/badge/API%20Provider/InstantAPI.ai/blue)](https://web.instantapi.ai) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Web Crawling Basics](https://badgen.net/badge/Notebook/Web%20Crawling%20Basics/orange)](notebooks/web_crawling_basics.ipynb) | Learn to build a basic web crawler using Scrapy |
| [![Distributed Data Collection](https://badgen.net/badge/Notebook/Distributed%20Data%20Collection/orange)](notebooks/distributed_data_collection.ipynb) | Set up a distributed crawling system with Kafka |
| [![Stream Processing Pipeline](https://badgen.net/badge/Notebook/Stream%20Processing%20Pipeline/orange)](notebooks/stream_processing_pipeline.ipynb) | Process real-time data streams with Spark |

### Data Cleaning & Preprocessing Pipelines
#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Data Cleaning with Python](https://badgen.net/badge/Tutorial/Data%20Cleaning%20with%20Python/blue)](https://www.kaggle.com/learn/data-cleaning) | |
| [![Text Preprocessing Techniques](https://badgen.net/badge/Blog/Text%20Preprocessing%20Techniques/pink)](https://towardsdatascience.com/8-steps-to-master-data-preparation-with-python-85555d45f54b) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![spaCy](https://badgen.net/badge/Framework/spaCy/green)](https://spacy.io/) | |
| [![NLTK](https://badgen.net/badge/Framework/NLTK/green)](https://www.nltk.org/) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Text Cleaning Pipeline](https://badgen.net/badge/Notebook/Text%20Cleaning%20Pipeline/orange)](notebooks/text_cleaning_pipeline.ipynb) | Build an end-to-end text cleaning pipeline |
| [![Advanced Preprocessing](https://badgen.net/badge/Notebook/Advanced%20Preprocessing/orange)](notebooks/advanced_preprocessing.ipynb) | Implement advanced text preprocessing techniques |

### Pre-training Datasets
- **Description**: Explore and utilize large-scale datasets suitable for pre-training language models, focusing on diverse, high-quality text corpora.
- **Concepts Covered**: `web crawling`, `data curation`, `quality filtering`, `deduplication`, `content diversity`, `multilingual data`, `domain-specific corpora`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![RedPajama Data Processing Guide](https://badgen.net/badge/Github%20Repository/RedPajama%20Data%20Processing%20Guide/cyan)](https://github.com/togethercomputer/RedPajama-Data) | [![Building High-Quality Pre-training Corpora](https://badgen.net/badge/Paper/Building%20High-Quality%20Pre-training%20Corpora/purple)](https://arxiv.org/abs/2010.12741) |
| [![The Pile: An 800GB Dataset of Diverse Text](https://badgen.net/badge/Website/The%20Pile/blue)](https://pile.eleuther.ai/) | [![SlimPajama Technical Report](https://badgen.net/badge/Paper/SlimPajama%20Technical%20Report/purple)](https://arxiv.org/abs/2401.07608) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Datasets-CLI](https://badgen.net/badge/Github%20Repository/Datasets-CLI/cyan)](https://github.com/huggingface/datasets-cli) | [![CCNet Processing Tools](https://badgen.net/badge/Github%20Repository/CCNet%20Processing%20Tools/cyan)](https://github.com/facebookresearch/cc_net) |
| [![FastText Language Detection](https://badgen.net/badge/Framework/FastText%20Language%20Detection/green)](https://fasttext.cc/docs/en/language-identification.html) | |
| [![Deduplicate-text-datasets](https://badgen.net/badge/Github%20Repository/Deduplicate-text-datasets/cyan)](https://github.com/google-research/deduplicate-text-datasets) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Dataset Creation Pipeline](https://badgen.net/badge/Notebook/Dataset%20Creation%20Pipeline/orange)](notebooks/dataset_creation_pipeline.ipynb) | Build an end-to-end dataset creation pipeline |
| [![Data Quality Assessment](https://badgen.net/badge/Notebook/Data%20Quality%20Assessment/orange)](notebooks/data_quality_assessment.ipynb) | Implement filtering and quality metrics |
| [![Deduplication Workshop](https://badgen.net/badge/Notebook/Deduplication%20Workshop/orange)](notebooks/deduplication_workshop.ipynb) | Practice text deduplication techniques |

#### Popular Datasets
| Dataset | Description |
|----------|-------------|
| [![The Pile](https://badgen.net/badge/Hugging%20Face%20Dataset/The%20Pile/yellow)](https://pile.eleuther.ai/) | 800GB dataset of diverse English text |
| [![RedPajama](https://badgen.net/badge/Hugging%20Face%20Dataset/RedPajama/yellow)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | 1T token dataset modeled after LLaMA training data |
| [![SlimPajama](https://badgen.net/badge/Hugging%20Face%20Dataset/SlimPajama/yellow)](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | Curated 627B token subset of RedPajama |
| [![C4](https://badgen.net/badge/Hugging%20Face%20Dataset/C4/yellow)](https://huggingface.co/datasets/c4) | Cleaned version of Common Crawl |
| [![ROOTS](https://badgen.net/badge/Hugging%20Face%20Dataset/ROOTS/yellow)](https://huggingface.co/datasets/bigscience-data/roots) | Multilingual dataset used to train BLOOM |
| [![PubMed Central](https://badgen.net/badge/Website/PubMed%20Central/blue)](https://www.ncbi.nlm.nih.gov/pmc/) | Biomedical and life sciences literature |
| [![ArXiv Dataset](https://badgen.net/badge/Hugging%20Face%20Dataset/ArXiv%20Dataset/yellow)](https://huggingface.co/datasets/arxiv_dataset) | Scientific papers from arXiv |
| [![GitHub Code](https://badgen.net/badge/Hugging%20Face%20Dataset/GitHub%20Code/yellow)](https://huggingface.co/datasets/codeparrot/github-code) | Programming code from GitHub repositories |
| [![mC4](https://badgen.net/badge/Hugging%20Face%20Dataset/mC4/yellow)](https://huggingface.co/datasets/mc4) | Multilingual version of C4 dataset |
| [![OSCAR](https://badgen.net/badge/Hugging%20Face%20Dataset/OSCAR/yellow)](https://huggingface.co/datasets/oscar) | Large-scale multilingual dataset |