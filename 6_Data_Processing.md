# Module 6: Data Processing

### Data Collection
#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [Common Crawl Documentation](https://commoncrawl.org/the-data/get-started/) | [Best Scraping Tools Directory](https://bestscrapingtools.com/web-crawling-tools/) |
| [Distributed Web Scraping Guide](https://www.scrapingbee.com/blog/distributed-web-scraping/) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [Common Crawl](https://commoncrawl.org/) | [Internet Archive](https://archive.org/web/) |
| [Scrapy](https://scrapy.org/) | [Colly](https://github.com/gocolly/colly) |
| [Apache Kafka](https://kafka.apache.org/) | [Spider-rs](https://github.com/spider-rs/spider) |
| [Apache Spark](https://spark.apache.org/) | [InstantAPI.ai](https://web.instantapi.ai) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| Web Crawling Basics | Learn to build a basic web crawler using Scrapy |
| Distributed Data Collection | Set up a distributed crawling system with Kafka |
| Stream Processing Pipeline | Process real-time data streams with Spark |

### Data Cleaning & Preprocessing Pipelines
#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [Data Cleaning with Python](https://www.kaggle.com/learn/data-cleaning) | |
| [Text Preprocessing Techniques](https://towardsdatascience.com/8-steps-to-master-data-preparation-with-python-85555d45f54b) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [spaCy](https://spacy.io/) | |
| [NLTK](https://www.nltk.org/) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| Text Cleaning Pipeline | Build an end-to-end text cleaning pipeline |
| Advanced Preprocessing | Implement advanced text preprocessing techniques |

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
| Dataset Creation Pipeline | Build an end-to-end dataset creation pipeline |
| Data Quality Assessment | Implement filtering and quality metrics |
| Deduplication Workshop | Practice text deduplication techniques |

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