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

### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![BPE Implementation](https://badgen.net/badge/Colab%20Notebook/BPE%20Implementation/orange)](https://colab.research.google.com/drive/1RwrtINbHTPBSRIoW8Zn9BRabxXguRRf0?usp=sharing) | Build a basic Byte Pair Encoding tokenizer from scratch |
| [![Hugging Face Tokenizers](https://badgen.net/badge/Colab%20Notebook/HF%20Tokenizers/orange)](https://colab.research.google.com/drive/1mcFgQ9PX1TFyEAsFOnoS1ozeSz3vM6A1?usp=sharing) | Learn to use Hugging Face tokenizers for text preparation |
| [![Custom Tokenizer](https://badgen.net/badge/Colab%20Notebook/Custom%20Tokenizer/orange)](https://colab.research.google.com/drive/1uYFoxwCKwshkchBgQ4y4z9cDfKRlwZ-e?usp=sharing) | Create and train a domain-specific tokenizer |
| [![New Tokenizer Training](https://badgen.net/badge/Colab%20Notebook/New%20Tokenizer%20Training/orange)](https://colab.research.google.com/drive/1452WFn66MZzYylTNcL6hV5Zd45sskzs7?usp=sharing) | Learn to train a new tokenizer from an existing one |
| [![GPT Tokenizer](https://badgen.net/badge/Colab%20Notebook/GPT%20Tokenizer/orange)](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing) | Build a BPE tokenizer from scratch based on GPT models |
| [![Tokenizer Comparison](https://badgen.net/badge/Colab%20Notebook/Tokenizer%20Comparison/orange)](https://colab.research.google.com/drive/1wVSCBGFm7KjJy-KugYGYETpncWsPgx5N?usp=sharing) | Compare custom tokenizers with state-of-the-art competitors |

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


## 2. Word Embeddings & Contextual Representations: Capturing Meaning in Vectors

Word embeddings are fundamental to Natural Language Processing, serving as dense vector representations of words that capture semantic and syntactic relationships.  Initially, techniques focused on static word embeddings, where each word had a fixed representation regardless of context. However, the field has significantly evolved, leading to more sophisticated methods that generate contextualized representations, adapting to the surrounding text to capture nuanced meanings.

**Key Concepts**

To understand the landscape of word embeddings, it's helpful to categorize them and understand the core ideas:

- **Static Word Embeddings:**
    - **Word Embeddings:** The general concept of representing words as vectors.
    - **Word2Vec Models (Skip-gram & CBOW):**  Early models that learn embeddings by predicting context words (Skip-gram) or a word from its context (CBOW).
    - **GloVe Embeddings (Global Vectors for Word Representation):** Embeddings learned from global word co-occurrence statistics, capturing broader context.
    - **Sparse Representations:**  Traditional one-hot encoding, which is high-dimensional and doesn't capture semantic similarity, contrasted with dense embeddings.
    - **Cosine Similarity:** A common metric to measure the semantic similarity between word vectors.

- **Contextual Word Embeddings:**
    - **Contextual Embeddings:** Word representations that vary based on the context in which they appear, capturing polysemy and nuanced meanings.
    - **Transformer-based Models (BERT, etc.):**  Models leveraging Transformer architectures and attention mechanisms to generate context-aware embeddings.
    - **Vector Representations:**  The output of embedding models, which are numerical vectors representing words or text.
    - **Semantic Similarity:**  Measuring how alike the meaning of words or phrases are, often enhanced by contextual embeddings.
    - **Word Embedding Models:**  Algorithms and architectures designed to create word embeddings.

- **Advanced Concepts:**
    - **Subword Information (FastText):**  Utilizing character n-grams to handle rare words and out-of-vocabulary terms, improving embeddings for morphologically rich languages.
    - **Multilingual and Cross-lingual Embeddings (Multilingual BERT):**  Embeddings designed to work across multiple languages, facilitating cross-lingual NLP tasks.
    - **Domain-Specific Embeddings:**  Tailoring embeddings to specific domains to capture specialized vocabulary and semantic relationships.
    - **Bias Detection and Mitigation:**  Addressing and reducing biases present in word embeddings to ensure fairness and equity in NLP applications.

### Essential Learning Sources

| Source | Description |
|--------|-------------|
| [![Word Embeddings Deep Dive](https://badgen.net/badge/Blog/Word%20Embeddings%20Deep%20Dive/pink)](https://lilianweng.github.io/posts/2017-10-15-word-embedding/) | **Comprehensive Overview:** This blog post provides an in-depth exploration of various word embedding techniques, including both static and contextual methods, along with implementation details. |
| [![CS224N Lecture 1 - Intro & Word Vectors](https://badgen.net/badge/Video/CS224N%20Lecture%201%20-%20Intro%20&%20Word%20Vectors/red)](https://www.youtube.com/watch?v=rmVRLeJRkl4) | **Foundational Knowledge:** Stanford's CS224N lecture offers a comprehensive introduction to word vectors, covering distributional semantics, Word2Vec algorithms (Skip-gram, CBOW), and optimization techniques. It's a great starting point for understanding the basics. |
| [![Illustrated Word2Vec](https://badgen.net/badge/Blog/Illustrated%20Word2Vec/pink)](https://jalammar.github.io/illustrated-word2vec/) | **Visual Learning:** This blog post provides a visually intuitive guide to understanding the Word2Vec model, making it easier to grasp the underlying mechanisms. |
| [![Contextual Embeddings](https://badgen.net/badge/Paper/Contextual%20Embeddings/purple)](https://www.cs.princeton.edu/courses/archive/spring20/cos598C/lectures/lec3-contextualized-word-embeddings.pdf) | **Deep Dive into Context:** Princeton's lecture notes delve into the concept of contextual embeddings, explaining their significance and applications in capturing dynamic word meanings. |
| [![Training Sentence Transformers](https://badgen.net/badge/Blog/Training%20Sentence%20Transformers/pink)](https://huggingface.co/blog/train-sentence-transformers) | **Practical Guide to Modern Embeddings:** This blog post offers a practical guide to training and fine-tuning sentence embedding models using Sentence Transformers v3, showcasing a modern approach to embedding learning. |
| [![BERT Paper](https://badgen.net/badge/Paper/BERT%20Paper/purple)](https://arxiv.org/abs/2204.03503) | **Revolutionizing Context:**  The original BERT paper is essential for understanding the architecture that significantly advanced contextual word embeddings. It introduces the Transformer-based approach and its impact on NLP. |
| [![GloVe Paper](https://badgen.net/badge/Paper/GloVe%20Paper/purple)](https://www.semanticscholar.org/paper/67b692bbfd29c5a30cfd1046efd5f85eecd1ea86) | **Global Context Matters:** The GloVe paper details how global word-word co-occurrence statistics are leveraged to learn embeddings, providing a different perspective compared to Word2Vec's local context window. |
| [![FastText Paper](https://badgen.net/badge/Paper/FastText%20Paper/purple)](https://www.semanticscholar.org/paper/d23e59abcae6ba653ba45dcc0ef975438890a3a4) | **Subword Embeddings:** The FastText paper introduces the concept of subword information in word embeddings, which is crucial for handling rare words and improving performance in morphologically rich languages. |

### Additional Learning Sources

| Source | Description |
|--------|-------------|
| [![Oddly Satisfying Deep Learning: Word Embeddings](https://badgen.net/badge/Book/Oddly%20Satisfying%20Deep%20Learning/blue)](https://pythonandml.github.io/dlbook/content/word_embeddings/traditional_word_embeddings.html) | **Intuitive Explanations:** This book chapter offers an intuitive introduction to word embeddings with visual aids and step-by-step code examples, making it accessible for beginners. |
| [![Instructor Embeddings](https://badgen.net/badge/Guide/Instructor%20Embeddings/blue)](https://huggingface.co/hkunlp/instructor-large) | **Task-Specific Embeddings:** Learn about task-specific embeddings with HuggingFace's Instructor, demonstrating how embeddings can be tailored for specific NLP tasks. |
| [![Custom LLM Embedding Training](https://badgen.net/badge/Tutorial/Custom%20LLM%20Embedding%20Training/blue)](https://dagshub.com/blog/how-to-train-a-custom-llm-embedding-model/) | **Hands-on Training:** A step-by-step guide on training custom LLM embedding models, providing practical insights into creating your own embeddings. |
| [![Word2Vec Implementation](https://badgen.net/badge/Tutorial/Word2Vec%20NumPy/blue)](https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/) | **Implementation from Scratch:**  A tutorial on implementing Word2Vec using Python and NumPy, ideal for understanding the inner workings of the algorithm. |
| [![Fruit Fly Word Embeddings](https://badgen.net/badge/Paper/Fruit%20Fly%20Embeddings/purple)](https://arxiv.org/abs/2101.06887) | **Novel Approaches:** Explore biologically-inspired sparse binary word embeddings based on the fruit fly brain, showcasing innovative research in the field. |
| [![Probabilistic FastText](https://badgen.net/badge/Paper/Probabilistic%20FastText/purple)](https://arxiv.org/abs/1806.02901) | **Advanced Techniques:**  Learn about multi-sense word embeddings that combine subword structure with uncertainty modeling, extending the capabilities of FastText. |
| [![Word Embeddings Guide](https://badgen.net/badge/Guide/Word%20Embeddings%20Guide/blue)](https://www.turing.com/kb/guide-on-word-embeddings-in-nlp) | **Comprehensive Guide:** A broad guide covering word embeddings from basic concepts to advanced techniques, offering a wide-ranging overview. |
| [![Multilingual BERT Paper](https://badgen.net/badge/Paper/Multilingual%20BERT%20Paper/purple)](https://www.semanticscholar.org/paper/0b0bc70b48aebe608d53a955990cb08f73de5a7d) | **Cross-Lingual Embeddings:**  The Multilingual BERT paper [2] introduces embeddings that work across different languages, enabling cross-lingual transfer learning and multilingual applications. |
| [![Bias in Contextualized Word Embeddings](https://badgen.net/badge/Paper/Bias%20in%20Contextualized%20Embeddings/purple)](https://www.semanticscholar.org/paper/5ea2104a039921633f75a9f4b986b515ddbe96d7) | **Bias Awareness:** This paper [5] highlights the issue of bias in contextualized word embeddings, particularly ethnic bias in news corpora, raising awareness about fairness in NLP. |

### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Gensim](https://badgen.net/badge/Framework/Gensim/green)](https://radimrehurek.com/gensim/) | [![FastText](https://badgen.net/badge/Framework/FastText/green)](https://fasttext.cc/) |
| [![Transformers](https://badgen.net/badge/Framework/Transformers/green)](https://huggingface.co/transformers/) | [![TensorFlow Text](https://badgen.net/badge/Framework/TensorFlow%20Text/green)](https://www.tensorflow.org/text) |
| [![Sentence Transformers](https://badgen.net/badge/Framework/Sentence%20Transformers/green)](https://www.sbert.net/) | [![spaCy](https://badgen.net/badge/Framework/spaCy/green)](https://spacy.io/) |

### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Word2Vec Implementation](https://badgen.net/badge/Colab%20Notebook/Word2Vec%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink3) | **Implement Word2Vec from scratch:**  Gain a deep understanding of Word2Vec by building it from the ground up. |
| [![GloVe Implementation](https://badgen.net/badge/Colab%20Notebook/GloVe%20Implementation/orange)](https://colab.research.google.com/drive/yournotebooklink4) | **Implement GloVe from scratch:**  Understand GloVe embeddings by implementing them yourself, focusing on global co-occurrence statistics. |
| [![Sentence Transformer Fine-tuning](https://badgen.net/badge/Colab%20Notebook/Sentence%20Transformer%20Fine-tuning/orange)](https://colab.research.google.com/drive/yournotebooklink_sentence_transformers) | **Fine-tune Sentence Transformers:** Learn to fine-tune pre-trained Sentence Transformer models for specific tasks, leveraging contextual embeddings. *(Replace `yournotebooklink_sentence_transformers` with an actual notebook link)* |
| [![BERT Embeddings Exploration](https://badgen.net/badge/Colab%20Notebook/BERT%20Embeddings%20Exploration/orange)](https://colab.research.google.com/drive/yournotebooklink_bert_exploration) | **Explore BERT Embeddings:**  Experiment with pre-trained BERT models to extract and analyze contextual word embeddings. *(Replace `yournotebooklink_bert_exploration` with an actual notebook link)* |



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
