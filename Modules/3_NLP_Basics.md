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
| [![GPT Tokenizer Implementation from Scratch](https://badgen.net/badge/Video/GPT-2%20Implementation%20from%20Scratch/red)](https://www.youtube.com/watch?v=kCc8FmEb1nY) | An optional but valuable tutorial on implementing GPT tokenizer from andrej karpathy. |
| [![Tokenization Fundamentals](https://badgen.net/badge/Course/Tokenization%20Fundamentals/orange)](https://huggingface.co/learn/nlp-course/chapter2/4) |Comprehensive course that covers tokenization basics, algorithms and best practices. |
| [![Stanford CS224N: Subword Models](https://badgen.net/badge/Course/Stanford%20CS224N%20Subword%20Models/orange)](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf) | Academic material providing a deep-dive into subword tokenization theory and applications. |

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
| [![BPE Implementation](https://badgen.net/badge/Colab%20Notebook/BPE%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink1) | Build a basic Byte Pair Encoding tokenizer from scratch |
| [![Custom Tokenizer](https://badgen.net/badge/Colab%20Notebook/Custom%20Tokenizer/orange)](https://colab.research.google.com/drive/yournotebooklink2) | Create and train a domain-specific tokenizer |


## 2. Word Embeddings & Contextual Representations

Learn techniques for representing words as vectors to capture semantic and syntactic relationships.

**Key Concepts**
- Word Embeddings
- Word2Vec Models
- GloVe Embeddings
- Contextual Embeddings
- Vector Representations
- Semantic Similarity

### Essential Learning Sources

| Source | Description |
|--------|-------------|
| [![Illustrated Word2Vec](https://badgen.net/badge/Blog/Illustrated%20Word2Vec/pink)](https://jalammar.github.io/illustrated-word2vec/) | Visual guide to understanding Word2Vec embeddings |

### Additional Learning Sources

| Source | Description |
|--------|-------------|
| [![GloVe Project](https://badgen.net/badge/Website/GloVe%20Project/blue)](https://nlp.stanford.edu/projects/glove/) | Stanford's GloVe project documentation and resources |

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
