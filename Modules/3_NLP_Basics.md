# Module 3: NLP Fundamentals

![image](https://github.com/user-attachments/assets/a64a3a2a-c65f-4dba-8338-4ff048636d45)

## Overview
This module covers essential Natural Language Processing concepts and techniques, focusing on text processing, word representations, and language modeling fundamentals crucial for understanding LLMs.

## Core Topics
### 1. Tokenization Strategies

Learn various tokenization methods to convert text into model-readable tokens.

#### Key Concepts
- Byte Pair Encoding (BPE)
- WordPiece Tokenization
- Unigram Tokenization
- Custom Tokenizers
- Domain-specific Tokenization
- Vocabulary Optimization

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Tokenization Fundamentals](https://badgen.net/badge/Course/Tokenization%20Fundamentals/orange)](https://huggingface.co/learn/nlp-course/chapter2/4) | [![SentencePiece Training Guide](https://badgen.net/badge/Docs/SentencePiece%20Training%20Guide/green)](https://github.com/google/sentencepiece#train-sentencepiece-model) |
| [![Stanford CS224N: Subword Models](https://badgen.net/badge/Course/Stanford%20CS224N%20Subword%20Models/orange)](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes05-LM_RNN.pdf) | [![GPT-2 Implementation from Scratch](https://badgen.net/badge/Video/GPT-2%20Implementation%20from%20Scratch/red)](https://www.youtube.com/watch?v=kCc8FmEb1nY) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Tokenizers](https://badgen.net/badge/Framework/Hugging%20Face%20Tokenizers/green)](https://huggingface.co/docs/tokenizers/index) | [![SmolGPT Implementation](https://badgen.net/badge/Github%20Repository/SmolGPT/cyan)](https://github.com/Om-Alve/smolGPT) |
| [![SentencePiece](https://badgen.net/badge/Github%20Repository/SentencePiece/cyan)](https://github.com/google/sentencepiece) | [![llama2.c Repository](https://badgen.net/badge/Github%20Repository/llama2.c/cyan)](https://github.com/karpathy/llama2.c) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![BPE Implementation](https://badgen.net/badge/Colab%20Notebook/BPE%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink1) | Build a basic Byte Pair Encoding tokenizer from scratch |
| [![Custom Tokenizer](https://badgen.net/badge/Colab%20Notebook/Custom%20Tokenizer/orange)](https://colab.research.google.com/drive/yournotebooklink2) | Create and train a domain-specific tokenizer |

### 2. Word Embeddings & Contextual Representations

Learn techniques for representing words as vectors to capture semantic and syntactic relationships.

#### Key Concepts
- Word Embeddings
- Word2Vec Models
- GloVe Embeddings
- Contextual Embeddings
- Vector Representations
- Semantic Similarity

#### Learning Sources

| Essential | Optional |
|-----------|----------|
| [![Illustrated Word2Vec](https://badgen.net/badge/Blog/Illustrated%20Word2Vec/pink)](https://jalammar.github.io/illustrated-word2vec/) | [![GloVe Project](https://badgen.net/badge/Website/GloVe%20Project/blue)](https://nlp.stanford.edu/projects/glove/) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Gensim](https://badgen.net/badge/Framework/Gensim/green)](https://radimrehurek.com/gensim/) | [![FastText](https://badgen.net/badge/Framework/FastText/green)](https://fasttext.cc/) |
| [![Transformers](https://badgen.net/badge/Framework/Transformers/green)](https://huggingface.co/transformers/) | [![TensorFlow Text](https://badgen.net/badge/Framework/TensorFlow%20Text/green)](https://www.tensorflow.org/text) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Word2Vec Implementation](https://badgen.net/badge/Colab%20Notebook/Word2Vec%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink3) | Implement Word2Vec from scratch |
| [![GloVe Implementation](https://badgen.net/badge/Colab%20Notebook/GloVe%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink4) | Implement GloVe from scratch |

### 3. Language Modeling Basics

Understand fundamental concepts of statistical language modeling and sequence prediction.

#### Key Concepts
- Language Modeling
- N-gram Models
- Probabilistic Models
- Next-word Prediction
- Model Architecture
- Training Approaches

#### Learning Sources

| Essential | Optional |
|-----------|----------|
| [![N-Gram Language Modeling Guide](https://badgen.net/badge/Tutorial/N-Gram%20Language%20Modeling%20Guide/blue)](https://www.geeksforgeeks.org/n-gram-language-modeling/) | [![Stanford CS224N](https://badgen.net/badge/Course/Stanford%20CS224N/orange)](https://web.stanford.edu/class/cs224n/) |
| [![Dense LLM Lecture](https://badgen.net/badge/Video/Dense%20LLM%20Lecture/red)](https://youtu.be/9vM4p9NN0Ts) | [![Stanford CS229](https://badgen.net/badge/Course/Stanford%20CS229/orange)](https://cs229.stanford.edu/) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![KenLM](https://badgen.net/badge/Framework/KenLM/green)](https://kheafield.com/code/kenlm/) | [![SRILM](https://badgen.net/badge/Framework/SRILM/green)](http://www.speech.sri.com/projects/srilm/) |
| [![PyTorch](https://badgen.net/badge/Framework/PyTorch/green)](https://pytorch.org/) | [![TensorFlow](https://badgen.net/badge/Framework/TensorFlow/green)](https://www.tensorflow.org/) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![N-Gram Language Modeling](https://badgen.net/badge/Colab%20Notebook/N-Gram%20Language%20Modeling/orange)](https://colab.research.google.com/drive/yournotebooklink5) | Implement N-Gram Language Modeling |
| [![Probabilistic Language Modeling](https://badgen.net/badge/Colab%20Notebook/Probabilistic%20Language%20Modeling/orange)](https://colab.research.google.com/drive/yournotebooklink6) | Implement Probabilistic Language Modeling |
