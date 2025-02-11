# Module 3: NLP Fundamentals

This module covers essential Natural Language Processing concepts and techniques, focusing on text processing, word representations, and language modeling fundamentals crucial for understanding LLMs.

## 1. Tokenization Strategies

Learn various tokenization methods to convert text into model-readable tokens.

**Key Concepts**
- Byte Pair Encoding (BPE)
- WordPiece Tokenization
- Unigram Tokenization
- Custom Tokenizers
- Domain-specific Tokenization
- Vocabulary Optimization

### Essential Learning Sources

| Source | Description |
|--------|-------------|
| [![GPT Tokenizer Implementation from Scratch](https://badgen.net/badge/Video/GPT-2%20Implementation%20from%20Scratch/red)](https://www.youtube.com/watch?v=zduSFxRajkE&t=4341s) | An optional but valuable tutorial on implementing GPT tokenizer from andrej karpathy. |
| [![Tokenization Fundamentals](https://badgen.net/badge/Course/Tokenization%20Fundamentals/orange)](https://huggingface.co/learn/nlp-course/chapter2/4) | HuggingFace course that covers tokenization basics, algorithms and best practices. |
| [![Stanford's CoreNLP: Tokenization](https://badgen.net/badge/Course/Stanford%20CS224N%20Subword%20Models/orange)](https://stanfordnlp.github.io/CoreNLP/tokenize.html) | Academic material providing a deep-dive into tokenization theory. |

### Additional Learning Sources

| Source | Description |
|--------|-------------|
| [![SentencePiece Training Guide](https://badgen.net/badge/Docs/SentencePiece%20Training%20Guide/green)](https://github.com/google/sentencepiece#train-sentencepiece-model) | A supplementary detailed guide on training custom SentencePiece models. |
| [![Tokenizer Shrinking Guide](https://badgen.net/badge/Guide/Tokenizer%20Shrinking%20Techniques/blue)](https://github.com/stas00/ml-engineering/blob/master/transformers/make-tiny-models.md) | Comprehensive guide on various tokenizer shrinking techniques |

### Tools

| Category | Tool | Description |
|----------|------|-------------|
| Playground | [![TikTokenizer](https://badgen.net/badge/Playground/TikTokenizer/blue)](https://tiktokenizer.vercel.app/) [![Hugging Face Tokenizer](https://badgen.net/badge/Playground/HF%20Tokenizer/blue)](https://huggingface.co/spaces/Xenova/the-tokenizer-playground) [![OpenAI Tokenizer](https://badgen.net/badge/Playground/OpenAI%20Tokenizer/blue)](https://platform.openai.com/tokenizer) [![Tokenizer Arena](https://badgen.net/badge/Playground/Tokenizer%20Arena/blue)](https://huggingface.co/spaces/Cognitive-Lab/Tokenizer_Arena) | Interactive visualization and experimentation |
| Library | [![Hugging Face Tokenizers](https://badgen.net/badge/Library/HF%20Tokenizers/green)](https://github.com/huggingface/tokenizers) [![SentencePiece](https://badgen.net/badge/Library/SentencePiece/green)](https://github.com/google/sentencepiece) [![Tiktoken](https://badgen.net/badge/Library/Tiktoken/green)](https://github.com/openai/tiktoken) [![spaCy](https://badgen.net/badge/Library/spaCy/green)](https://spacy.io/) [![Mistral Tokenizer](https://badgen.net/badge/Library/Mistral%20Tokenizer/green)](https://docs.mistral.ai/guides/tokenization/) | Production-ready tokenization implementation |

### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![BPE Implementation](https://badgen.net/badge/Colab%20Notebook/BPE%20Implementation/orange)](https://colab.research.google.com/drive/1RwrtINbHTPBSRIoW8Zn9BRabxXguRRf0?usp=sharing) | Build a basic Byte Pair Encoding tokenizer from scratch |
| [![Hugging Face Tokenizers](https://badgen.net/badge/Colab%20Notebook/HF%20Tokenizers/orange)](https://colab.research.google.com/drive/1mcFgQ9PX1TFyEAsFOnoS1ozeSz3vM6A1?usp=sharing) | Learn to use Hugging Face tokenizers for text preparation |
| [![Custom Tokenizer](https://badgen.net/badge/Colab%20Notebook/Custom%20Tokenizer/orange)](https://colab.research.google.com/drive/1uYFoxwCKwshkchBgQ4y4z9cDfKRlwZ-e?usp=sharing) | Create and train a domain-specific tokenizer |
| [![New Tokenizer Training](https://badgen.net/badge/Colab%20Notebook/New%20Tokenizer%20Training/orange)](https://colab.research.google.com/drive/1452WFn66MZzYylTNcL6hV5Zd45sskzs7?usp=sharing) | Learn to train a new tokenizer from an existing one |
| [![GPT Tokenizer](https://badgen.net/badge/Colab%20Notebook/GPT%20Tokenizer/orange)](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing) | Build a BPE tokenizer from scratch based on GPT models |
| [![Tokenizer Comparison](https://badgen.net/badge/Colab%20Notebook/Tokenizer%20Comparison/orange)](https://colab.research.google.com/drive/1wVSCBGFm7KjJy-KugYGYETpncWsPgx5N?usp=sharing) | Compare custom tokenizers with state-of-the-art competitors |


## 2. Word Embeddings & Contextual Representations

Learn techniques for representing words as vectors to capture semantic and syntactic relationships.

**Key Concepts**
- Word Embeddings
- Word2Vec Models
- GloVe Embeddings
- Contextual Embeddings
- Vector Representations
- Semantic Similarity
- Sparse Representations
- Cosine Similarity
- Word Embedding Models

### Essential Learning Sources

| Source | Description |
|--------|-------------|
| [![CS224N Lecture 1 - Intro & Word Vectors](https://badgen.net/badge/Video/CS224N%20Lecture%201%20-%20Intro%20&%20Word%20Vectors/red)](https://www.youtube.com/watch?v=rmVRLeJRkl4) | Comprehensive introduction to word vectors, covering distributional semantics, word embeddings, Word2Vec algorithm and optimization techniques |
| [![Word Embeddings Lecture](https://badgen.net/badge/Lecture/Word%20Embeddings%20Social%20Science/orange)](https://lse-me314.github.io/lecturenotes/ME314_day12.pdf) | Comprehensive lecture covering word embeddings fundamentals, estimation, applications, bias analysis, and social science use cases |
| [![Illustrated Word2Vec](https://badgen.net/badge/Blog/Illustrated%20Word2Vec/pink)](https://jalammar.github.io/illustrated-word2vec/) | Visual guide to understanding Word2Vec embeddings |
| [![Vector Space Models](https://badgen.net/badge/Blog/Vector%20Space%20Models/pink)](https://ruder.io/word-embeddings-1/) | Sebastian Ruder's comprehensive blog on word embedding models |
| [![Word Embeddings Guide](https://badgen.net/badge/Tutorial/Word%20Embeddings%20Guide/blue)](https://www.tensorflow.org/text/guide/word_embeddings) | Comprehensive TensorFlow guide on implementing word embeddings |
| [![Word2Vec Tutorial](https://badgen.net/badge/Tutorial/Word2Vec%20Tutorial/orange)](https://www.cs.toronto.edu/~lczhang/360/lec/w05/w2v.html) | Detailed tutorial on Word2Vec architecture and implementation |
| [![Word2Vec Paper](https://badgen.net/badge/Paper/Word2Vec%20Original/purple)](https://arxiv.org/abs/1301.3781) | Original Word2Vec paper introducing skip-gram and CBOW models ||
| [![GloVe Paper](https://badgen.net/badge/Paper/GloVe%20Original/purple)](https://nlp.stanford.edu/pubs/glove.pdf) | Original GloVe paper on global word vector representations |


### Additional Learning Sources

| Source | Description |
|--------|-------------|
| [![Word Embeddings Deep Dive](https://badgen.net/badge/Blog/Word%20Embeddings%20Deep%20Dive/pink)](https://lilianweng.github.io/posts/2017-10-15-word-embedding/) | Comprehensive blog post covering embedding techniques and implementations |
| [![Contextual Embeddings](https://badgen.net/badge/Paper/Contextual%20Embeddings/purple)](https://www.cs.princeton.edu/courses/archive/spring20/cos598C/lectures/lec3-contextualized-word-embeddings.pdf) | Princeton's lecture on contextual embeddings and their applications |
| [![GloVe Project](https://badgen.net/badge/Website/GloVe%20Project/blue)](https://nlp.stanford.edu/projects/glove/) | Stanford's GloVe project documentation and resources |
| [![FastText Resources](https://badgen.net/badge/Website/FastText%20Resources/blue)](https://fasttext.cc/) | Official FastText documentation and pre-trained embeddings |
| [![Instructor Embeddings](https://badgen.net/badge/Guide/Instructor%20Embeddings/blue)](https://huggingface.co/hkunlp/instructor-large) | Guide to task-specific embeddings with HuggingFace |
| [![Gensim Tutorial](https://badgen.net/badge/Tutorial/Gensim%20Word2Vec/blue)](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html) | Tutorial on training word embeddings with Gensim |
| [![FastText Guide](https://badgen.net/badge/Guide/FastText%20Embeddings/blue)](https://fasttext.cc/docs/en/crawl-vectors.html) | Guide to using FastText for document embeddings and classification |
| [![Word2Vec Implementation](https://badgen.net/badge/Tutorial/Word2Vec%20NumPy/blue)](https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/) | Tutorial on implementing Word2Vec with Python and NumPy |
| [![Fruit Fly Word Embeddings](https://badgen.net/badge/Paper/Fruit%20Fly%20Embeddings/purple)](https://arxiv.org/abs/2101.06887) | Novel biologically-inspired sparse binary word embeddings based on fruit fly brain |
| [![Probabilistic FastText](https://badgen.net/badge/Paper/Probabilistic%20FastText/purple)](https://arxiv.org/abs/1806.02901) | Multi-sense word embeddings combining subword structure with uncertainty modeling |
| [![Word Embeddings Guide](https://badgen.net/badge/Guide/Word%20Embeddings%20Guide/blue)](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp) | Comprehensive guide covering word embeddings from basics to advanced concepts |

### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Gensim](https://badgen.net/badge/Framework/Gensim/green)](https://radimrehurek.com/gensim/) | [![FastText](https://badgen.net/badge/Framework/FastText/green)](https://fasttext.cc/) |
| [![Transformers](https://badgen.net/badge/Framework/Transformers/green)](https://huggingface.co/transformers/) | [![TensorFlow Text](https://badgen.net/badge/Framework/TensorFlow%20Text/green)](https://www.tensorflow.org/text) |

### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Word2Vec Implementation](https://badgen.net/badge/Colab%20Notebook/Word2Vec%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink3) | Implement Word2Vec from scratch |
| [![GloVe Implementation](https://badgen.net/badge/Colab%20Notebook/GloVe%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink4) | Implement GloVe from scratch |


## 3. Language Modeling Basics

Understand fundamental concepts of statistical language modeling and sequence prediction.

**Key Concepts**
- Language Modeling
- N-gram Models
- Probabilistic Models
- Next-word Prediction
- Model Architecture
- Training Approaches

### Essential Learning Sources

| Source | Description |
|--------|-------------|
| [![N-Gram Language Modeling Guide](https://badgen.net/badge/Tutorial/N-Gram%20Language%20Modeling%20Guide/blue)](https://www.geeksforgeeks.org/n-gram-language-modeling/) | Comprehensive guide to N-Gram language modeling |
| [![Dense LLM Lecture](https://badgen.net/badge/Video/Dense%20LLM%20Lecture/red)](https://youtu.be/9vM4p9NN0Ts) | In-depth lecture on dense language models |

### Additional Learning Sources

| Source | Description |
|--------|-------------|
| [![Stanford CS224N](https://badgen.net/badge/Course/Stanford%20CS224N/orange)](https://web.stanford.edu/class/cs224n/) | Advanced NLP course from Stanford |
| [![Stanford CS229](https://badgen.net/badge/Course/Stanford%20CS229/orange)](https://cs229.stanford.edu/) | Machine Learning fundamentals course |

### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![KenLM](https://badgen.net/badge/Framework/KenLM/green)](https://kheafield.com/code/kenlm/) | [![SRILM](https://badgen.net/badge/Framework/SRILM/green)](http://www.speech.sri.com/projects/srilm/) |
| [![PyTorch](https://badgen.net/badge/Framework/PyTorch/green)](https://pytorch.org/) | [![TensorFlow](https://badgen.net/badge/Framework/TensorFlow/green)](https://www.tensorflow.org/) |

### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![N-Gram Language Modeling](https://badgen.net/badge/Colab%20Notebook/N-Gram%20Language%20Modeling/orange)](https://colab.research.google.com/drive/yournotebooklink5) | Implement N-Gram Language Modeling |
| [![Probabilistic Language Modeling](https://badgen.net/badge/Colab%20Notebook/Probabilistic%20Language%20Modeling/orange)](https://colab.research.google.com/drive/yournotebooklink6) | Implement Probabilistic Language Modeling |
