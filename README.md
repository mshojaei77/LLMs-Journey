# Mastering Large Language Models: From Foundations to Production

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

---

## Course Overview

This course provides a comprehensive guide to Large Language Models (LLMs), covering everything from foundational concepts to production deployment. Each module includes practical tasks, additional resources, and relevant research papers to deepen your understanding and skills.

## Table of Contents

- [Module 0: Essential Foundations for LLM Development](#module-0-essential-foundations-for-llm-development)
- [Module 1: Introduction to Large Language Models](#module-1-introduction-to-large-language-models)
- [Module 2: Transformer Architecture Details](#module-2-transformer-architecture-details)
- [Module 3: Data Preparation and Tokenization](#module-3-data-preparation-and-tokenization)
- [Module 4: Building an LLM from Scratch: Core Components](#module-4-building-an-llm-from-scratch-core-components)
- [Module 5: Pretraining LLMs](#module-5-pretraining-llms)
- [Module 6: Evaluating LLMs](#module-6-evaluating-llms)
- [Module 7: Core LLM Architectures (High-Level)](#module-7-core-llm-architectures-high-level)
- [Module 8: Training & Optimization](#module-8-training--optimization)
- [Module 9: Evaluation & Validation](#module-9-evaluation--validation)
- [Module 10: Fine-tuning & Adaptation](#module-10-fine-tuning--adaptation)
- [Module 11: Inference Optimization](#module-11-inference-optimization)
- [Module 12: Deployment & Scaling](#module-12-deployment--scaling)
- [Module 13: Advanced Applications](#module-13-advanced-applications)
- [Module 14: Ethics & Security](#module-14-ethics--security)
- [Module 15: Maintenance & Monitoring](#module-15-maintenance--monitoring)
- [Module 16: Multimodal Systems](#module-16-multimodal-systems)
- [Module 17: Capstone Project](#module-17-capstone-project)
- [Module 18: Emerging Trends](#module-18-emerging-trends)

---

## Module 0: Essential Foundations for LLM Development

**Objective:** Establish the fundamental mathematical and computational knowledge required for understanding and developing LLMs.

- [x] **Linear Algebra Fundamentals for LLMs** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nCNL7Ro5vOPWS5yaqTMpz2B056TyjsHy?usp=sharing)
  - **Description:** Covers essential linear algebra concepts like vectors, matrices, matrix operations, and their relevance to neural networks and LLMs.
  - **Additional Resource**: [Essence of Linear Algebra" series by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

- [x] **Probability Foundations for LLMs** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oFu5ZL0AUlxDU8xhygr-datwEnHS9JVN?usp=sharing)
  - **Description:** Introduces probability theory, distributions, and statistical concepts crucial for understanding language models and their probabilistic nature.
  - **Additional Resource**: [An Intuitive Guide to How LLMs Work" by J. Lowin](https://www.jlowin.dev/blog/an-intuitive-guide-to-how-llms-work)

- [x] **GPU Essentials for LLMs** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1S-LwgyO_bmS135nJmJxm1ZKVlpv9Acfv?usp=sharing)
  - **Description:** Provides an overview of GPU architecture, CUDA programming, and best practices for efficient computation in deep learning, specifically for training and deploying LLMs.
  - **Additional Resource**: [Ultimate Guide to the Best NVIDIA GPUs for Running Large Language Models" from Spheron Network](https://blog.spheron.network/ultimate-guide-to-the-best-nvidia-gpus-for-running-large-language-models)

---
## Module 1: Introduction to Large Language Models

**Objective:** Define LLMs, explore their history, capabilities, limitations, and understand their significance in the field of NLP and AI.

- [ ] **Overview of LLMs** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Introduces the concept of Large Language Models, their scale, and their impact on Natural Language Processing.
    - **Podcast:** [Building an LLM Twin: From Concept to Deployment](https://notebooklm.google.com/notebook/ad84858b-63a7-4ac8-8dc1-b8cc224b7df9/audio)
    - **Additional Sources:**
        - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
        - [Stanford NLP Deep Learning Guide](https://web.stanford.edu/~jurafsky/slp3/)
    - **Papers:**
        - [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
        - [Language Models are Few-Shot Learners (2020)](https://arxiv.org/abs/2005.14165)

- [ ] **The "Large" in LLMs: Scale and Implications** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explores the "large" aspect of LLMs in terms of parameters, datasets, and computational resources, and discusses the implications of scale.
    - **Practical Tasks:**
        1. Calculate parameter count vs training data size ratios for different LLMs.
        2. Compare memory requirements for 7B vs 70B parameter models.
        3. Research compute costs for training modern LLMs.
    - **Additional Sources:**
        - [AI and Compute (OpenAI)](https://openai.com/research/ai-and-compute)
        - [LLM Scaling Laws](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications)
    - **Papers:**
        - [Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361)
        - [Chinchilla's Wild Implications (2022)](https://arxiv.org/abs/2203.15556)

- [ ] **Capabilities of LLMs: What Can They Do?** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Examines the diverse capabilities of LLMs, including text generation, translation, question answering, and more, highlighting emergent abilities.
    - **Practical Tasks:**
        1. Implement zero-shot classification with prompt engineering.
        2. Compare few-shot vs fine-tuning performance on a custom dataset.
        3. Build a multilingual translation system using a single LLM.
    - **Additional Sources:**
        - [Emergent Abilities of LLMs](https://arxiv.org/abs/2206.07682)
        - [HuggingFace Tasks Guide](https://huggingface.co/docs/transformers/tasks)
    - **Papers:**
        - [Language Models are Multitask Learners (2019)](https://arxiv.org/abs/1910.10683)
        - [Beyond the Imitation Game (2023)](https://arxiv.org/abs/2206.04615)

- [ ] **Limitations of LLMs: Challenges and Pitfalls** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Discusses the limitations of LLMs, such as hallucinations, biases, knowledge cut-off, and ethical concerns.
    - **Practical Tasks:**
        1. Generate and analyze hallucination examples from different LLMs.
        2. Test temporal knowledge cutoff dates for various models.
        3. Evaluate bias in model outputs across demographic groups.
    - **Additional Sources:**
        - [AI Safety Fundamentals](https://aisafetyfundamentals.com/)
        - [Model Cards Toolkit](https://modelcards.withgoogle.com/about)
    - **Papers:**
        - [TruthfulQA: Measuring How Models Mimic Human Falsehoods (2021)](https://arxiv.org/abs/2109.07958)
        - [Taxonomy of Risks from Language Models (2022)](https://arxiv.org/abs/2207.07411)

- [ ] **Historical Context and Evolution of LLMs** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Traces the historical development of NLP and language models leading up to modern LLMs, highlighting key milestones and paradigm shifts.
    - **Practical Tasks:**
        1. Create an interactive timeline of NLP milestones from 1950s to present.
        2. Compare Word2Vec vs Transformer-based word representations.
        3. Conduct interview-style Q&A about paradigm shifts in AI and NLP.
    - **Additional Sources:**
        - [AI Timeline](https://www.assemblyai.com/blog/the-full-story-of-large-language-models-and-rlhf/)
        - [Deep Learning NLP History](https://ruder.io/a-review-of-the-recent-history-of-nlp/)
    - **Papers:**
        - [Distributed Representations of Words (2013)](https://arxiv.org/abs/1301.3781)
        - [BERT: Pre-training of Deep Bidirectional Transformers (2018)](https://arxiv.org/abs/1810.04805)

---

## Module 2: Transformer Architecture Details

**Objective:** Deep dive into the Transformer architecture, understanding its components and their functionalities.

- [ ] **Encoder-Decoder Architecture** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvardnlp/annotated-transformer/blob/master/The%20Annotated%20Transformer.ipynb)
    - **Description:** Detailed exploration of the encoder-decoder structure, its application in sequence-to-sequence tasks, and its relevance to early Transformer models.
    - **Practical Tasks:**
        1. Implement an encoder stack with N=6 identical layers in PyTorch.
        2. Build a decoder with a masked self-attention mechanism.
        3. Train a basic neural machine translation (EN->DE) model.
    - **Additional Sources:**
        - [Transformers from Scratch](https://e2eml.school/transformers.html)
        - [Transformer Basics Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
        - [Sequence-to-Sequence Modeling](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html)
    - **Papers:**
        - [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
        - [Efficient Transformers: A Survey (2020)](https://arxiv.org/abs/2009.06732)

- [ ] **Decoder-Only Models** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-)
    - **Description:** Focus on decoder-only architectures like GPT, their advantages for text generation, and the concept of causal attention.
    - **Practical Tasks:**
        1. Implement causal attention masking in a Transformer decoder block.
        2. Fine-tune GPT-2 for creative story generation.
        3. Compare decoder-only vs encoder-decoder model performance on text generation tasks.
    - **Additional Sources:**
        - [HuggingFace Generation Docs](https://huggingface.co/docs/transformers/generation_strategies)
        - [LLM University - Decoders](https://llm.university/)
    - **Papers:**
        - [Language Models are Unsupervised Multitask Learners (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
        - [LLaMA: Open and Efficient Foundation Models (2023)](https://arxiv.org/abs/2302.13971)

- [ ] **Self-Attention Mechanism** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rPk3ohrmVclqhH7uQ7qys4oznDdAhpzF)
    - **Description:** In-depth understanding of the self-attention mechanism, its mathematical formulation, and its role in capturing relationships within sequences.
    - **Practical Tasks:**
        1. Compute attention scores from given Q/K/V matrices manually and programmatically.
        2. Visualize attention patterns for different sentence structures using attention visualization tools.
        3. Implement relative position encoding in the self-attention mechanism.
    - **Additional Sources:**
        - [Attention Visualization Tool (BertViz)](https://github.com/jessevig/bertviz)
        - [Math of Self-Attention (deeplearning.ai Short Course)](https://deeplearning.ai/short-courses/mathematics-of-transformer/)
    - **Papers:**
        - [Self-Attention with Relative Position (2018)](https://arxiv.org/abs/1803.02155)
        - [Longformer: Local+Global Attention (2020)](https://arxiv.org/abs/2004.05150)

- [ ] **Multi-Head Attention** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/examples/blob/main/transformer/transformer_tutorial.ipynb)
    - **Description:** Exploration of multi-head attention, its benefits in capturing diverse relationships, and implementation details.
    - **Practical Tasks:**
        1. Implement parallel attention heads within a Transformer block.
        2. Analyze head specialization patterns in pre-trained Transformer models.
        3. Experiment with different numbers of attention heads (4, 8, 12, 16) and observe performance changes.
    - **Additional Sources:**
        - [Multi-Head Attention Explained (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
        - [Attention Head Analysis](https://arxiv.org/abs/2005.00753)
    - **Papers:**
        - [Are Sixteen Heads Really Better Than One? (2019)](https://arxiv.org/abs/1905.10650)
        - [Talking Heads Attention (2020)](https://arxiv.org/abs/2003.02436)

- [ ] **Positional Encoding** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rPk3ohrmVclqhH7uQ7qys4oznDdAhpzF)
    - **Description:** Understanding the necessity of positional encoding, different encoding methods (sinusoidal, learned, etc.), and their impact on sequence modeling.
    - **Practical Tasks:**
        1. Implement sinusoidal positional encoding as described in the original Transformer paper.
        2. Compare sinusoidal encoding with learned positional embeddings in a simple task.
        3. Experiment with ALiBi (Attention with Linear Biases) positional encoding.
    - **Additional Sources:**
        - [Positional Encoding Explorer](https://github.com/jalammar/positional-encoding-explorer)
        - [Rotary Embeddings Guide (RoPE)](https://blog.eleuther.ai/rotary-embeddings/)
    - **Papers:**
        - [RoPE: Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)
        - [Train Short, Test Long (2021)](https://arxiv.org/abs/2108.12409)

- [ ] **Feed-Forward Networks in Transformers** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/text_classification_with_transformer.ipynb)
    - **Description:** Examination of the Feed-Forward Network (FFN) within each Transformer layer, its role in non-linearity and feature transformation.
    - **Practical Tasks:**
        1. Implement a position-wise Feed-Forward Network as used in Transformers.
        2. Experiment with different activation functions within FFNs (GELU vs ReLU) and analyze performance.
        3. Study the impact of hidden layer dimension in FFNs on model performance and size.
    - **Additional Sources:**
        - [FFN Architecture Analysis in Transformers](https://arxiv.org/abs/2205.05638)
        - [GELU Activation Paper](https://arxiv.org/abs/1606.08415)
    - **Papers:**
        - [GLU Variants Improve Transformer (2022)](https://arxiv.org/abs/2002.05202)
        - [DeepNet: Scaling Transformers (2022)](https://arxiv.org/abs/2203.00555)

- [ ] **Layer Normalization in Transformers** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p6vS3hHxJWzaQOxv8rXZh8zBk1jVpW-M)
    - **Description:** Understanding Layer Normalization, its placement in Transformer blocks, and its importance for training stability and performance.
    - **Practical Tasks:**
        1. Compare pre-Layer Normalization vs post-Layer Normalization architectures in terms of training stability.
        2. Implement RMSNorm (Root Mean Square Layer Normalization) as an alternative to LayerNorm.
        3. Debug gradient flow issues and see how normalization helps in deep Transformer networks.
    - **Additional Sources:**
        - [Normalization Deep Dive: Layer Normalization](https://leimao.github.io/blog/Layer-Normalization/)
        - [DeepNorm Implementation (DeepSpeed)](https://github.com/microsoft/DeepSpeed/tree/master/deepspeed)
    - **Papers:**
        - [Root Mean Square Layer Normalization (2019)](https://arxiv.org/abs/1910.07467)
        - [Understanding LN in Transformers (2020)](https://arxiv.org/abs/2002.04745)

- [ ] **Residual Connections in Transformers** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/fairseq/blob/main/examples/translation/README.md)
    - **Description:** Importance of residual connections (skip connections) in deep networks, particularly in Transformers, for enabling gradient flow and training deep models.
    - **Practical Tasks:**
        1. Build a Transformer block with residual/skip connections and verify its structure.
        2. Analyze gradient magnitudes across layers with and without residual connections.
        3. Experiment with different weights for residual connections and observe training dynamics.
    - **Additional Sources:**
        - [Residual/Skip Connections Theory Explained](https://theaisummer.com/skip-connections/)
        - [Transformer Residuals Paper (Analyzing Residual Stream)](https://arxiv.org/abs/2305.14864)
    - **Papers:**
        - [ResNet: Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385) *(Original ResNet paper)*
        - [DeepNet: Scaling Transformers to 1,000 Layers (2022)](https://arxiv.org/abs/2203.00555)

---

## Module 3: Data Preparation and Tokenization

**Objective:** Learn the crucial steps of data collection, preprocessing, and tokenization necessary for training and utilizing LLMs effectively.

- [ ] **Data Collection Strategies for LLM Training** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Methods for gathering large text datasets, including web scraping, public datasets, and ethical considerations in data collection.
    - **Practical Tasks:**
        1. Build a web scraping pipeline with politeness controls (rate limiting, user-agent management).
        2. Implement data deduplication techniques at scale to ensure data quality.
        3. Create a bias evaluation report for a collected dataset to understand potential biases.
    - **Additional Sources:**
        - [The Pile Dataset Paper](https://arxiv.org/abs/2101.00027)
        - [Data Governance for Machine Learning](https://datagovernance.org/)
    - **Papers:**
        - [Deduplicating Training Data Makes Language Models Better (2021)](https://arxiv.org/abs/2107.06499)
        - [Red Teaming Language Models to Reduce Harms (2022)](https://arxiv.org/abs/2202.03286)

- [x] **Tokenization Exploration: BPE, WordPiece, Unigram** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mshojaei77/NLP-Journey/blob/main/ch1/Tokenization_BPE.ipynb)
    - **Description:** Understanding different tokenization algorithms used in LLMs, including Byte Pair Encoding (BPE), WordPiece, and Unigram, and their trade-offs.
    - **Additional Sources:**
        - ["The Technical User's Introduction to LLM Tokenization" by Christopher Samiullah](https://christophergs.com/blog/understanding-llm-tokenization)
        - [Byte Pair Encoding (BPE) Visual Guide (Video Tutorial)](https://www.youtube.com/watch?v=HEikzVL-lZU)
        - [Tokenizers: How Machines Read (Interactive Guide)](https://lena-voita.github.io/nlp_course/tokenization.html)
    - **Papers:**
        - [Neural Machine Translation of Rare Words with Subword Units (2016)](https://arxiv.org/abs/1508.07909) *(Original BPE Paper)*
        - [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates (2018)](https://arxiv.org/abs/1804.10959) *(Unigram Tokenization)*

- [x] **Hugging Face Tokenizers Library** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mshojaei77/NLP-Journey/blob/main/ch1/Hugging_Face_Tokenizers.ipynb)
    - **Description:** Practical application of the Hugging Face `tokenizers` library for efficient tokenization and understanding its functionalities.
    - **Additional Sources:**
        - [Advanced Tokenization Strategies (Hugging Face Video Guide)](https://www.youtube.com/watch?v=VFp38yj8h3A)
        - [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer)
        - [Tokenization for Multilingual Models (Hugging Face Course)](https://huggingface.co/course/chapter6/2?fw=pt)
    - **Papers:**
        - [BERT: Pre-training of Deep Bidirectional Transformers (2019)](https://arxiv.org/abs/1810.04805) *(WordPiece in BERT)*
        - [How Good is Your Tokenizer? Evaluating Tokenization Strategies for Pre-trained Language Models (2021)](https://aclanthology.org/2021.emnlp-main.571.pdf) *(Tokenizer Evaluation)*

- [x] **Training Custom Tokenizers** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uYFoxwCKwshkchBgQ4y4z9cDfKRlwZ-e?usp=sharing)
    - **Description:** Step-by-step guide on training a custom tokenizer from scratch using a given dataset, tailoring tokenization to specific domains or languages.
    - **Additional Sources:**
        - [Train a Tokenizer for Code (Andrej Karpathy’s "Let’s Build the GPT Tokenizer")](https://www.youtube.com/watch?v=zduSFxRajkE) *(Video Tutorial)*
        - [Domain-Specific Tokenizers with SentencePiece](https://github.com/google/sentencepiece/blob/master/README.md)
        - [Tokenizer Best Practices (Hugging Face Docs)](https://huggingface.co/docs/tokenizers/quicktour#training-a-new-tokenizer-from-an-old-one)
    - **Papers:**
        - [Getting the Most Out of Your Tokenizer: Pre-training Tokenizers for Neural Language Models (2024)](https://arxiv.org/html/2402.01035v2)
        - [TokenMonster: Towards Efficient Vocabulary-Sensitive Tokenization (2023)](https://arxiv.org/abs/2310.08946)

- [ ] **Embedding Techniques: Word, Sentence, and Positional Embeddings** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Covers different types of embeddings used in LLMs, including word embeddings, sentence embeddings, and positional embeddings, and their roles in representing text and sequence information.
    - **Additional Sources:**
        - [Embeddings Explained Visually (Word2Vec)](https://jalammar.github.io/illustrated-word2vec/)
        - [Positional Encoding in Transformers Explained](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
    - **Papers:**
        - [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) *(Positional Embeddings)*
        - [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (2019)](https://arxiv.org/abs/1908.10084)

- [ ] **Text Vectorization Methods** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explores methods for converting text into numerical vectors, including TF-IDF, Bag-of-Words, and learned embeddings, and their applicability in different NLP tasks.
    - **Practical Tasks:**
        1. Implement TF-IDF vectorization and compare it with learned word embeddings for text similarity tasks.
        2. Visualize embedding spaces using dimensionality reduction techniques like PCA and t-SNE.
        3. Build hybrid sparse+dense representations for text data.
    - **Additional Sources:**
        - [Embeddings Guide (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)
        - [Sentence Transformers Library](https://www.sbert.net/)
    - **Papers:**
        - [Efficient Estimation of Word Representations in Vector Space (2013)](https://arxiv.org/abs/1301.3781) *(Word2Vec)*
        - [BERT Rediscovers the Classical NLP Pipeline (2019)](https://arxiv.org/abs/1905.05950)

- [ ] **Data Preprocessing and Cleaning for LLMs** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Techniques for cleaning and preprocessing text data, including handling noise, special characters, normalization, and preparing data for LLM training.
    - **Practical Tasks:**
        1. Create a multilingual text cleaning pipeline to handle various text formats and languages.
        2. Implement parallel processing techniques for efficient preprocessing of large datasets.
        3. Build quality classifiers to filter out low-quality or irrelevant text data.
    - **Additional Sources:**
        - [Text Processing Best Practices (Stanford NLP Book)](https://nlp.stanford.edu/IR-book/html/htmledition/text-processing-1.html)
        - [Unicode Normalization Explained](https://unicode.org/reports/tr15/)
    - **Papers:**
        - [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data (2020)](https://arxiv.org/abs/1911.00359)
        - [The Curse of Low-Quality Training Data: An Empirical Study of Pretraining Data Quality for Language Models (2022)](https://arxiv.org/abs/2205.11487)

---

## Module 4: Building an LLM from Scratch: Core Components

**Objective:** Guide through the process of building an LLM from the ground up, focusing on implementing core components using PyTorch.

- [ ] **Coding a Minimal LLM in PyTorch** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Step-by-step coding of a basic Transformer-based LLM, focusing on core functionalities and simplicity for educational purposes.
    - **Practical Tasks:**
        1. Initialize model weights using PyTorch and understand initialization strategies.
        2. Implement a basic forward pass through a simplified Transformer model.
        3. Profile memory usage across different layers of the model to understand resource consumption.
    - **Additional Sources:**
        - [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
        - [Model Memory Footprint Calculator](https://modelmemory.com/)
    - **Papers:**
        - [GPT in 60 Lines of NumPy (by Jay Mody) (2023)](https://jaykmody.com/blog/gpt-from-scratch/) *(Conceptual simplicity)*
        - [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (2020)](https://arxiv.org/abs/1909.08053) *(For understanding scale)*

- [ ] **Implementation of Transformer Layers** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Detailed implementation of key Transformer layers, including multi-head attention, positional embeddings, and feed-forward networks in PyTorch.
    - **Practical Tasks:**
        1. Code a multi-head attention layer from scratch in PyTorch, ensuring efficiency.
        2. Implement rotary positional embeddings (RoPE) and integrate them into the model.
        3. Add dropout regularization to different parts of the Transformer model for improved generalization.
    - **Additional Sources:**
        - [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) *(Code walkthrough)*
        - [Flash Attention Implementation (HazyResearch)](https://github.com/HazyResearch/flash-attention) *(For efficiency ideas)*
    - **Papers:**
        - [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)](https://arxiv.org/abs/2205.14135) *(Efficient attention mechanisms)*
        - [ALiBi: Train Short, Test Long with Exponentially Scaled Attention (2021)](https://arxiv.org/abs/2108.12409) *(Alternative positional encoding)*

- [ ] **Layer Normalization and Gradient Management** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implementing Layer Normalization and techniques for managing gradients, such as gradient clipping, to stabilize training of deep LLMs.
    - **Practical Tasks:**
        1. Compare pre-normalization vs post-normalization Transformer architectures empirically.
        2. Implement gradient clipping with norm awareness to prevent exploding gradients.
        3. Debug common issues related to exploding or vanishing gradients in deep networks.
    - **Additional Sources:**
        - [Normalization Techniques Explained](https://leimao.github.io/blog/Layer-Normalization/)
        - [PyTorch Normalization Layers Documentation](https://pytorch.org/docs/stable/nn.html#normalization-layers)
    - **Papers:**
        - [Understanding Deep Learning Requires Rethinking Generalization (2017)](https://arxiv.org/abs/1611.03530) *(Generalization in deep learning)*
        - [On Layer Normalization in the Transformer Architecture (2020)](https://arxiv.org/abs/2002.04745) *(In-depth analysis of LayerNorm)*

- [ ] **Parameter Initialization and Management** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Strategies for efficient parameter management in large models, including parameter initialization, sharding, and memory optimization.
    - **Practical Tasks:**
        1. Implement basic parameter sharding across multiple GPUs for model parallelism.
        2. Profile GPU memory usage during model initialization and training.
        3. Create a mixed-precision training configuration to reduce memory footprint and speed up training.
    - **Additional Sources:**
        - [Model Parallelism Guide (Hugging Face)](https://huggingface.co/docs/transformers/parallelism)
        - [GPU Memory Management in PyTorch](https://pytorch.org/docs/stable/notes/cuda.html)
    - **Papers:**
        - [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2020)](https://arxiv.org/abs/1910.02054) *(Memory optimization techniques)*
        - [8-bit Optimizers via Block-wise Quantization (2022)](https://arxiv.org/abs/2110.02861) *(Memory-efficient optimizers)*

---

## Module 5: Pretraining LLMs

**Objective:** Cover the process of pretraining LLMs, including methodologies, objectives, and practical considerations.

- [ ] **Pretraining Data and Process** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Overview of the LLM pretraining process, datasets used (e.g., The Pile), and the typical workflow.
    - **Practical Tasks:**
        1. Set up distributed data loading for large pretraining datasets.
        2. Implement masked language modeling (MLM) objective for BERT-style pretraining.
        3. Monitor key training dynamics using tools like Weights & Biases (WandB).
    - **Additional Sources:**
        - [HuggingFace Pretraining Guide](https://huggingface.co/docs/transformers/training)
        - [MLOps for Pretraining Pipelines](https://ml-ops.org/)
    - **Papers:**
        - [RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)](https://arxiv.org/abs/1907.11692) *(Pretraining optimizations)*
        - [The Pile: A 800GB Dataset of Diverse Text for Language Modeling (2020)](https://arxiv.org/abs/2101.00027) *(Pretraining dataset)*

- [ ] **Next-Word Prediction and Language Modeling** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Focus on next-word prediction as the core pretraining task for generative LLMs, and related concepts like perplexity.
    - **Practical Tasks:**
        1. Implement causal attention masks to enforce autoregressive next-word prediction.
        2. Compare different loss functions for language modeling (Cross-Entropy vs Focal Loss).
        3. Analyze prediction confidence of a pretrained model across different text domains.
    - **Additional Sources:**
        - [Language Modeling Basics Explained](https://lena-voita.github.io/nlp_course/language_modeling.html)
        - [Perplexity in Language Models Explained](https://towardsdatascience.com/perplexity-in-language-models-87a196019a94)
    - **Papers:**
        - [Improving Language Understanding by Generative Pre-Training (OpenAI GPT-1) (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
        - [Scaling Laws for Autoregressive Generative Language Modeling (2020)](https://arxiv.org/abs/2001.08361) *(Scaling laws in pretraining)*

- [ ] **Self-Supervised Learning Objectives** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Understanding self-supervised learning (SSL) as the paradigm for LLM pretraining, and exploring different SSL objectives beyond language modeling.
    - **Practical Tasks:**
        1. Design and implement contrastive learning objectives for text representations.
        2. Implement data augmentation techniques relevant to self-supervised text learning.
        3. Evaluate the quality of learned representations using probing tasks.
    - **Additional Sources:**
        - [Self-Supervised Learning: Methods and Applications Survey (2019)](https://arxiv.org/abs/1902.06162)
        - [Self-Supervised Learning for Speech (wav2vec 2.0)](https://ai.meta.com/blog/wav2vec-2-0-learning-the-structure-of-speech-from-raw-audio/) *(Example from another domain)*
    - **Papers:**
        - [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (2020)](https://arxiv.org/abs/2006.11477) *(SSL example)*
        - [Emerging Properties in Self-Supervised Learning (2021)](https://arxiv.org/abs/2104.14294) *(Emergence in SSL)*

- [ ] **Training Loop and Optimization Strategies** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Setting up an efficient and stable training loop for LLMs, including gradient accumulation, learning rate schedules, and checkpointing.
    - **Practical Tasks:**
        1. Implement gradient accumulation to train with larger effective batch sizes.
        2. Add learning rate warmup and decay schedules to the training process.
        3. Set up a robust model checkpointing strategy to save progress and enable recovery.
    - **Additional Sources:**
        - [PyTorch Lightning Training Loops](https://lightning.ai/docs/pytorch/stable/common/optimization.html)
        - [Guide to Training Stability and Avoiding Exploding Gradients (WandB)](https://wandb.ai/site/articles/how-to-avoid-exploding-gradients)
    - **Papers:**
        - [Adam: A Method for Stochastic Optimization (2015)](https://arxiv.org/abs/1412.6980) *(Adam optimizer)*
        - [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (2018)](https://arxiv.org/abs/1804.04235) *(Memory-efficient optimizer)*

- [ ] **Computational Costs and Infrastructure for Pretraining** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Understanding the computational demands of LLM pretraining, estimating costs, and considering infrastructure choices (cloud providers, hardware).
    - **Practical Tasks:**
        1. Estimate FLOPs (Floating Point Operations) for a given model architecture and training dataset.
        2. Compare cloud training costs across different providers (AWS, GCP, Azure) for LLM training.
        3. Explore and implement techniques for energy-efficient training.
    - **Additional Sources:**
        - [Machine Learning CO2 Impact Calculator](https://mlco2.github.io/impact/)
        - [Efficient Machine Learning Book](https://efficientml.ai/)
    - **Papers:**
        - [Green AI (2019)](https://arxiv.org/abs/1907.10597) *(Sustainability in AI)*
        - [The Computational Limits of Deep Learning (2020)](https://arxiv.org/abs/2007.05558) *(Computational constraints)*

- [ ] **Saving, Loading, and Sharing Pretrained Models** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Best practices for saving model checkpoints, loading pretrained models, and sharing models efficiently.
    - **Practical Tasks:**
        1. Implement incremental checkpointing to save model states periodically during long pretraining runs.
        2. Convert model formats between PyTorch and ONNX for interoperability.
        3. Set up automatic recovery mechanisms from training failures using saved checkpoints.
    - **Additional Sources:**
        - [PyTorch Checkpointing Mechanisms](https://pytorch.org/docs/stable/checkpoint.html)
        - [Model Serialization Best Practices (Safetensors)](https://huggingface.co/docs/safetensors/en/index)
    - **Papers:**
        - [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (2019)](https://arxiv.org/abs/1811.06965) *(Pipeline parallelism and checkpointing)*
        - [ZeRO-Offload: Democratizing Billion-Scale Model Training (2021)](https://arxiv.org/abs/2101.06840) *(Efficient training and checkpointing)*

---

## Module 6: Evaluating LLMs

**Objective:** Master the methods and metrics for evaluating LLMs, covering both automatic and human evaluation approaches.

- [ ] **Text Generation Metrics: BLEU, ROUGE, and Beyond** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Introduction to common automatic metrics for evaluating text generation quality, such as BLEU and ROUGE, and their limitations.
    - **Practical Tasks:**
        1. Implement BLEU and ROUGE scoring functions for evaluating generated text.
        2. Set up a human evaluation pipeline to assess text quality qualitatively.
        3. Analyze the trade-offs between diversity and quality in text generation and how metrics capture these.
    - **Additional Sources:**
        - [Survey of Evaluation Metrics for Natural Language Generation (NLG)](https://arxiv.org/abs/1612.09332)
        - [HuggingFace Evaluate Hub](https://huggingface.co/docs/evaluate/index) *(Library for evaluation metrics)*
    - **Papers:**
        - [BLEU: a Method for Automatic Evaluation of Machine Translation (2002)](https://aclanthology.org/P02-1040.pdf) *(Original BLEU paper)*
        - [ROUGE: A Package for Automatic Evaluation of Summaries (2004)](https://aclanthology.org/W04-1013.pdf) *(Original ROUGE paper)*

- [ ] **Importance of Comprehensive Evaluation** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Emphasizing the need for holistic and multi-faceted evaluation of LLMs, going beyond simple accuracy metrics to assess various aspects like bias, toxicity, and robustness.
    - **Practical Tasks:**
        1. Create an evaluation rubric tailored to specific domain-specific tasks for LLMs.
        2. Compare automated evaluation metrics with human evaluations for different aspects of LLM performance.
        3. Implement adversarial test cases to probe model robustness and identify failure modes.
    - **Additional Sources:**
        - [HELM: Holistic Evaluation of Language Models (Stanford CRFM)](https://crfm.stanford.edu/helm/latest/)
        - [BigBench: Beyond the Imitation Game? (Google)](https://github.com/google/BIG-bench) *(Benchmark tasks)*
    - **Papers:**
        - [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList (2020)](https://arxiv.org/abs/2005.04118) *(Behavioral testing)*
        - [Dynabench: Rethinking Benchmarking in NLP (2021)](https://arxiv.org/abs/2106.06052) *(Dynamic benchmarking)*

- [ ] **Loss Metrics and Training Dynamics Analysis** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Using loss metrics (training loss, validation loss) as indicators of model training progress and performance, and analyzing training dynamics.
    - **Practical Tasks:**
        1. Track and visualize training and validation loss curves to diagnose training issues.
        2. Implement custom loss functions to guide model training towards specific objectives.
        3. Analyze correlations between loss values and downstream task performance.
    - **Additional Sources:**
        - [Loss Landscape Visualization Tools](https://losslandscape.com/)
        - [PyTorch Loss Functions Documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)
    - **Papers:**
        - [An Empirical Study of Training Dynamics for Deep Neural Networks (2021)](https://arxiv.org/abs/2106.06934) *(Training dynamics analysis)*
        - [The Curse of Low Task Diversity: On the Generalization of Multi-task Learning (2022)](https://arxiv.org/abs/

---

## Module 7: Core LLM Architectures (High-Level)

**Objective:**  Provide a high-level overview of different LLM architectures beyond the basic Transformer, including Encoder, Decoder, and Hybrid models.

- [ ] **Self-Attention Mechanism: Deep Dive & Implementation** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Revisit self-attention with a focus on implementation details and advanced techniques.
    - **Practical Tasks:**
        - Build basic self-attention in PyTorch with attention weight visualization.
    - **Additional Sources**:
        - [Refer back to Module 2 resources on Self-Attention](#module-2-transformer-architecture-details)
    - **Papers**:
        - [Refer back to Module 2 papers on Self-Attention](#module-2-transformer-architecture-details)

- [ ] **Transformer Encoder Architecture** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Focus on the structure and function of the Transformer Encoder in models like BERT.
    - **Practical Tasks:**
        - Implement a Transformer Encoder with positional encoding and feed-forward networks.
    - **Additional Sources**:
        - [Refer back to Module 2 resources on Encoder-Decoder Architecture](#module-2-transformer-architecture-details)
    - **Papers**:
        - [BERT paper (refer back to Module 1 or 2)](#module-1-introduction-to-large-language-models) or [#module-2-transformer-architecture-details]

- [ ] **Multi-Head Attention: Advanced Applications** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explore advanced uses and variations of multi-head attention in different architectures.
    - **Practical Tasks:**
        - Compare performance of models with varying numbers of heads on sequence-to-sequence tasks.
    - **Additional Sources**:
        - [Refer back to Module 2 resources on Multi-Head Attention](#module-2-transformer-architecture-details)
    - **Papers**:
        - [Refer back to Module 2 papers on Multi-Head Attention](#module-2-transformer-architecture-details)

- [ ] **Normalization Techniques: Comparative Study** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:**  Compare and contrast different normalization techniques (LayerNorm, RMSNorm, BatchNorm) in LLMs.
    - **Practical Tasks:**
        - Experiment with LayerNorm vs RMSNorm impacts on training stability and performance.
    - **Additional Sources**:
        - [Refer back to Module 2 resources on Layer Normalization](#module-2-transformer-architecture-details)
    - **Papers**:
        - [Refer back to Module 2 papers on Layer Normalization](#module-2-transformer-architecture-details)

- [ ] **Residual Connections: In-depth Analysis** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:**  Analyze the impact of residual connections and explore variations in their implementation.
    - **Practical Tasks:**
        - Analyze architectures with and without residual connections and compare their training.
    - **Additional Sources**:
        - [Refer back to Module 2 resources on Residual Connections](#module-2-transformer-architecture-details)
    - **Papers**:
        - [Refer back to Module 2 papers on Residual Connections](#module-2-transformer-architecture-details)

---

## Module 8: Training & Optimization

**Objective:** Master modern training techniques for LLMs, focusing on efficiency and stability.

- [ ] **Mixed Precision Training**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implement and understand Mixed Precision Training for faster and memory-efficient training.
    - **Practical tasks:**
        - Implement AMP (Automatic Mixed Precision) with gradient scaling in PyTorch.
    - **Additional Sources**:
        - [PyTorch Automatic Mixed Precision (AMP) Documentation](https://pytorch.org/docs/stable/amp.html)
        - [NVIDIA Mixed Precision Training Guide](https://developer.nvidia.com/blog/accelerating-ai-training-with-automatic-mixed-precision/)
    - **Papers**:
        - [Mixed Precision Training (2017)](https://arxiv.org/abs/1710.03740)

- [ ] **LoRA Fine-tuning: Parameter-Efficient Adaptation**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Learn and implement LoRA (Low-Rank Adaptation) for efficient fine-tuning of LLMs.
    - **Practical Tasks:**
        - Adapt a pre-trained model (e.g., LLaMA-7B) for medical QA using the PubMedQA dataset with LoRA.
    - **Additional Sources**:
        - [LoRA for Parameter-Efficient Fine-Tuning of LLMs Blog Post](https://huggingface.co/blog/lora)
        - [Hugging Face PEFT Library Documentation](https://huggingface.co/docs/peft/index)
    - **Papers**:
        - [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685)

- [ ] **Distributed Training Strategies**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explore and implement distributed training techniques for scaling LLM training across multiple GPUs and machines.
    - **Practical Tasks:**
        - Configure PyTorch DDP (Distributed Data Parallel) to train a model across multiple GPUs.
    - **Additional Sources**:
        - [PyTorch Distributed Training Documentation](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
        - [Hugging Face Distributed Training Guide](https://huggingface.co/docs/transformers/v4.29.1/en/main_classes/trainer#distributed-training)
    - **Papers**:
        - [Efficient Large-Scale Language Model Training on GPU Clusters (2019)](https://arxiv.org/abs/1909.10305)

- [ ] **Hyperparameter Optimization for LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Learn and apply hyperparameter optimization techniques specifically for LLMs to find optimal configurations.
    - **Practical Tasks:**
        - Use Bayesian optimization (e.g., Optuna, Ray Tune) to optimize hyperparameters for an LLM.
    - **Additional Sources**:
        - [Optuna Documentation](https://optuna.org/)
        - [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
        - [Weights & Biases Hyperparameter Optimization Guide](https://wandb.ai/site/articles/hyperparameter-optimization)
    - **Papers**:
        - [Efficient Hyperparameter Optimization using Population Based Training and Bayesian Optimization (2017)](https://proceedings.mlr.press/v70/jaderberg17a.html)

- [ ] **Gradient Clipping and Accumulation Strategies**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implement and understand gradient clipping and accumulation strategies for training stability and larger batch sizes.
    - **Practical Tasks:**
        - Implement gradient clipping and gradient accumulation in a training loop for LLMs.
    - **Additional Sources**:
        - [Gradient Clipping Explained](https://machinelearningmastery.com/gradient-clipping-for-training-of-deep-neural-networks/)
        - [Gradient Accumulation for Deep Learning](https://kozodoi.me/blog/2021/02/19/gradient-accumulation.html)
    - **Papers**:
        - [On the importance of initialization and momentum in deep learning (2013)](http://proceedings.mlr.press/v28/sutskever13.pdf) (Discusses gradient issues in deep networks)

---

## Module 9: Evaluation & Validation

**Objective:** Build robust evaluation and validation systems for LLMs, focusing on different aspects of model quality.

- [ ] **Toxicity Detection and Mitigation**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Learn to detect and mitigate toxicity in LLM outputs, using tools and techniques for responsible AI.
    - **Practical Tasks:**
        - Create an ensemble toxicity detector using Perspective API and other methods.
    - **Additional Sources**:
        - [Perspective API by Google](https://perspectiveapi.com/)
        - [Hugging Face Detoxify Library](https://github.com/unitaryai/detoxify)
        - [RealToxicityPrompts Dataset](https://aclanthology.org/2020.emnlp-main.603/)
    - **Papers**:
        - [Challenges in Detoxifying Language Models (2021)](https://arxiv.org/abs/2101.00063)

- [ ] **Human Evaluation Platform Design**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Design and understand the components of a human evaluation platform for assessing LLM performance qualitatively.
    - **Practical Tasks:**
        - Build a basic web application for human evaluation, allowing for model comparisons and annotation.
    - **Additional Sources**:
        - [Amazon Mechanical Turk (AMT)](https://www.mturk.com/)
        - [Figure Eight (Appen)](https://appen.com/)
        - [Guidelines for Human Evaluation in NLP](https://aclanthology.org/W17-5501/)
    - **Papers**:
        - [Human Evaluation of Text Generation (2019)](https://aclanthology.org/W19-8601/)

- [ ] **Perplexity Analysis Across Datasets**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implement and analyze perplexity as an intrinsic evaluation metric across different datasets and domains.
    - **Practical Tasks:**
        - Implement perplexity calculation across different text datasets to analyze model fit.
    - **Additional Sources**:
        - [Perplexity Explained in Detail](https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3)
        - [Calculating Perplexity with Hugging Face Transformers](https://huggingface.co/docs/transformers/perplexity)
    - **Papers**:
        - [Evaluating Language Models (2013)](https://aclanthology.org/D13-1180/)

- [ ] **Bias Assessment and Fairness Metrics**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Learn to assess and measure bias in LLMs using fairness benchmarks and metrics.
    - **Practical Tasks:**
        - Measure fairness and bias using Hugging Face benchmarks and toolkits.
    - **Additional Sources**:
        - [Hugging Face Evaluate - Fairness Metrics](https://huggingface.co/docs/evaluate/fairness_metrics)
        - [Fairlearn Toolkit by Microsoft](https://fairlearn.org/v0.7.0/user_guide/index.html)
        - [Datasets for Bias Evaluation in NLP](https://ruder.io/nlp-bias/)
    - **Papers**:
        - [On Measuring and Mitigating Biases in NLP (2019)](https://aclanthology.org/W19-3829/)
        - [Fairness in Machine Learning (2018)](https://arxiv.org/abs/1812.00068)

---

## Module 10: Fine-tuning & Adaptation

**Objective:** Specialize pre-trained LLMs for specific downstream tasks and domains through fine-tuning.

- [ ] **Medical RAG System Development**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Build a Retrieval-Augmented Generation (RAG) system for medical question answering using PubMed data and fine-tuned LLMs.
    - **Practical Tasks:**
        - Develop a RAG system using PubMed-based retrieval and fine-tuned LLMs for medical queries.
    - **Additional Sources**:
        - [Hugging Face RAG Documentation](https://huggingface.co/docs/transformers/tasks/retrieval_augmentation)
        - [PubMed Dataset](https://pubmed.ncbi.nlm.nih.gov/)
        - [FAISS for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
    - **Papers**:
        - [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)](https://arxiv.org/abs/2005.11401)

- [ ] **Legal Document Analysis with Fine-tuned LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Apply fine-tuned LLMs for legal document analysis tasks, such as contract clause classification and information extraction.
    - **Practical Tasks:**
        - Fine-tune an LLM for contract clause classification in legal documents.
    - **Additional Sources**:
        - [CUAD Dataset for Contract Understanding](https://cuad.joelg.ai/)
        - [Legal-BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
        - [spaCy for Legal NLP](https://spacy.io/solutions/legal-nlp)
    - **Papers**:
        - [LegalBERT: A Pre-trained Language Model for the Legal Domain (2020)](https://arxiv.org/abs/2010.02559)

- [ ] **Parameter-Efficient Fine-Tuning (PEFT) Techniques**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Compare and implement Parameter-Efficient Fine-Tuning methods like LoRA, Adapters, and Prompt Tuning.
    - **Practical Tasks:**
        - Compare LoRA vs Adapters in terms of performance and efficiency for fine-tuning.
    - **Additional Sources**:
        - [Hugging Face PEFT Library](https://huggingface.co/docs/peft/index)
        - [Adapters for NLP: A Review](https://arxiv.org/abs/2005.00247)
    - **Papers**:
        - [Parameter-Efficient Transfer Learning for NLP (Adapters) (2019)](https://arxiv.org/abs/1902.00751)
        - [Prefix-Tuning: Optimizing Continuous Prompts for Generation (Prompt Tuning)](https://arxiv.org/abs/2101.00132)

- [ ] **Cross-Domain Adaptation and Fine-tuning**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explore strategies for cross-domain adaptation and fine-tuning of LLMs across different domains (medical, legal, tech, etc.).
    - **Practical Tasks:**
        - Fine-tune an LLM across multiple domains (e.g., medical, legal, tech) and analyze cross-domain performance.
    - **Additional Sources**:
        - [Domain Adaptation in NLP: A Survey](https://ruder.io/domain-adaptation/)
        - [Cross-Domain Few-Shot Learning via Meta-Learning](https://arxiv.org/abs/2003.04142)
    - **Papers**:
        - [Universal Language Model Fine-tuning for Text Classification (ULMFiT) (2018)](https://arxiv.org/abs/1801.06146) (Early work on transfer learning in NLP)

---

## Module 11: Inference Optimization

**Objective:** Enhance the efficiency of LLM inference to make models faster and more cost-effective for deployment.

- [ ] **KV-Cache Implementation for Faster Inference**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implement and understand KV-caching to accelerate LLM inference by reusing computed key and value vectors.
    - **Practical Tasks:**
        - Implement KV-Cache in a Transformer model to accelerate inference.
    - **Additional Sources**:
        - [KV-Caching Explained for Faster Inference](https://huggingface.co/docs/transformers/v4.29.1/en/perf_infer_gpu_fp16_accelerate)
        - [DeepSpeed Inference with KV-Cache](https://www.deepspeed.ai/tutorials/inference-tutorial/#kv-cache)
    - **Papers**:
        - [Reducing Transformer Sequence Length for Long-Document Classification (Mentions KV-Cache benefits)](https://arxiv.org/abs/2107.09386)

- [ ] **Quantization Techniques: 4-bit and Beyond**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explore and compare different quantization techniques (4-bit, 8-bit, GPTQ, AWQ) for reducing model size and accelerating inference.
    - **Practical Tasks:**
        - Compare GPTQ vs AWQ 4-bit quantization methods in terms of performance and memory reduction.
    - **Additional Sources**:
        - [GPTQ: Accurate Post-training Quantization for Generative Transformers](https://arxiv.org/abs/2210.17323)
        - [AWQ: Activation-aware Weight Quantization for LLMs](https://arxiv.org/abs/2306.00978)
        - [BitsAndBytes Library for Quantization](https://github.com/TimDettmers/bitsandbytes)
    - **Papers**:
        - [GPTQ: Accurate Post-training Quantization for Generative Transformers (2022)](https://arxiv.org/abs/2210.17323)
        - [AWQ: Activation-aware Weight Quantization for Large Language Models (2023)](https://arxiv.org/abs/2306.00978)

- [ ] **Model Pruning for Inference Speedup**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implement and understand model pruning techniques to remove less important weights and accelerate inference.
    - **Practical Tasks:**
        - Implement magnitude-based pruning for an LLM and measure inference speedup.
    - **Additional Sources**:
        - [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
        - [SparseML Library for Pruning](https://sparseml.neuralmagic.com/)
        - [Neural Magic Blog on Pruning](https://neuralmagic.com/blog/introduction-to-model-pruning-with-deepsparse-and-sparseml/)
    - **Papers**:
        - [Pruning Filters for Efficient ConvNets (2016)](https://arxiv.org/abs/1608.08710) (Early work on pruning, concepts applicable to Transformers)

- [ ] **Knowledge Distillation for Smaller Models**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Use knowledge distillation to train smaller, faster models that mimic the behavior of larger LLMs for efficient inference.
    - **Practical Tasks:**
        - Train a smaller Transformer model using knowledge distillation from a larger LLM.
    - **Additional Sources**:
        - [Knowledge Distillation Tutorial](https://towardsdatascience.com/knowledge-distillation-simplified-ddc070724770)
        - [Hugging Face DistilBERT Model (Example of Distillation)](https://huggingface.co/distilbert-base-uncased)
    - **Papers**:
        - [Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531) (Original Knowledge Distillation Paper)
        - [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (2019)](https://arxiv.org/abs/1910.01108) (Distillation example)

---

## Module 12: Deployment & Scaling

**Objective:** Learn about deploying and scaling LLMs for production environments, addressing infrastructure and cost considerations.

- [ ] **Kubernetes Orchestration for LLM Endpoints**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Deploy LLM inference endpoints using Kubernetes for auto-scaling, load balancing, and robust management.
    - **Practical Tasks:**
        - Deploy an LLM inference service on Kubernetes with auto-scaling capabilities.
    - **Additional Sources**:
        - [Kubernetes Documentation](https://kubernetes.io/docs/home/)
        - [Deploying Machine Learning Models with Kubernetes](https://www.kubeflow.org/docs/components/serving/) (Kubeflow Serving)
        - [Seldon Core for ML Model Deployment](https://www.seldon.io/)
    - **Papers**:
        - [Large-Scale Model Serving Infrastructure: Challenges and Solutions at Google (2021)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/49979.pdf) (Google's approach to model serving)

- [ ] **Security Hardening for LLM Applications**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implement security measures to protect LLM applications from vulnerabilities, including input/output sanitization and adversarial attacks.
    - **Practical Tasks:**
        - Implement input and output sanitization techniques for an LLM-powered application.
    - **Additional Sources**:
        - [OWASP Top Ten for LLM Applications](https://owasp.org/www-project-top-ten/) (When available, refer to OWASP or similar security guidelines for LLMs)
        - [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai/ai-risk-management-framework)
        - [Security in Machine Learning Course (Stanford)](https://cs359.stanford.edu/)
    - **Papers**:
        - [Adversarial Attacks on NLP: A Survey (2018)](https://arxiv.org/abs/1809.00987) (Understanding threats)

- [ ] **Edge Deployment of LLMs for Mobile and IoT Devices**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Optimize LLMs for deployment on edge devices with limited resources, such as mobile phones and IoT devices.
    - **Practical Tasks:**
        - Optimize an LLM for mobile inference and deploy it on a mobile simulator or device.
    - **Additional Sources**:
        - [TensorFlow Lite for Mobile and Edge Deployment](https://www.tensorflow.org/lite)
        - [ONNX Runtime for Edge Inference](https://onnxruntime.ai/)
        - [Qualcomm AI Engine for Edge AI](https://www.qualcomm.com/products/features/ai)
    - **Papers**:
        - [MobileBERT: A Compact Task-Agnostic BERT for Resource-Limited Devices (2020)](https://arxiv.org/abs/2004.02984) (Example of a compact BERT)

- [ ] **Cost Calculation and TCO Analysis for Cloud Deployment**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Analyze the costs associated with cloud deployment of LLMs, including compute, storage, and network costs, and perform Total Cost of Ownership (TCO) analysis.
    - **Practical Tasks:**
        - Perform a TCO analysis for deploying an LLM application on a cloud platform (AWS, GCP, Azure).
    - **Additional Sources**:
        - [AWS Pricing Calculator](https://calculator.aws/)
        - [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator)
        - [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
        - [Cloud Cost Management Tools](https://www.bmc.com/blogs/cloud-cost-management-tools/)
    - **Papers**:
        - [The Economics of Cloud Computing (2010)](https://static.googleusercontent.com/media/research.google.com/en//archive/jeffdean_wsc2010.pdf) (Foundational paper on cloud economics)

---

## Module 13: Advanced Applications

**Objective:** Explore cutting-edge applications of LLMs, pushing the boundaries of what's possible with these models.

- [ ] **Multimodal Assistant Development**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Build a multimodal assistant that integrates text and images, using models like CLIP and LLMs for tasks like image captioning and visual question answering.
    - **Practical Tasks:**
        - Develop a multimodal assistant with CLIP+GPT for image captioning.
    - **Additional Sources**:
        - [CLIP (Contrastive Language-Image Pre-training) by OpenAI](https://openai.com/research/clip)
        - [Hugging Face Transformers for Multimodal Models](https://huggingface.co/docs/transformers/multimodal)
        - [BLIP (Bootstrapping Language-Image Pre-training)](https://arxiv.org/abs/2201.05794)
    - **Papers**:
        - [Learning Transferable Visual Models From Natural Language Supervision (CLIP) (2021)](https://arxiv.org/abs/2103.00020)

- [ ] **Code Repair and Generation Engine**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Build an LLM-based code repair and generation engine to assist developers in debugging and writing code.
    - **Practical Tasks:**
        - Develop an LLM-based debugging tool that can suggest code repairs.
    - **Additional Sources**:
        - [GitHub Copilot](https://github.com/features/copilot)
        - [CodeBERT](https://huggingface.co/microsoft/codebert-base)
        - [CodeGen by Salesforce](https://blog.salesforceairesearch.com/codegen-code-generation-models/)
    - **Papers**:
        - [CodeBERT: A Pre-Trained Model for Programming and Natural Languages (2020)](https://arxiv.org/abs/2002.09436)

- [ ] **Personalized Tutor System with LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Design and prototype a personalized tutor system powered by LLMs, adapting to student needs and learning styles.
    - **Practical Tasks:**
        - Develop a prototype of an adaptive learning system using LLMs as a personalized tutor.
    - **Additional Sources**:
        - [Khan Academy using GPT-4 for Tutoring](https://blog.khanacademy.org/khan-academy-is-using-gpt-4-to-pilot-an-ai-tutor-for-math-and-more/)
        - [Carnegie Learning AI Tutor](https://www.carnegielearning.com/products/mathia-adventure/)
        - [EdTech and AI in Education](https://www.edsurge.com/research/reports/artificial-intelligence-in-education-2023)
    - **Papers**:
        - [AI-Powered Personalized Education (2018)](https://www.researchgate.net/publication/328174892_AI-Powered_Personalized_Education)

- [ ] **AI Red Teaming and Adversarial Attack Simulation**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Learn about AI red teaming and simulate adversarial attacks on LLMs to identify vulnerabilities and improve robustness.
    - **Practical Tasks:**
        - Conduct adversarial attack simulation on an LLM to identify vulnerabilities and failure points.
    - **Additional Sources**:
        - [AI Red Teaming Guide by MITRE](https://www.mitre.org/capabilities/cybersecurity/cybersecurity-innovation/adversarial-ml-threats/ai-red-teaming)
        - [Adversarial Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/)
        - [Robustness Gym](https://robustnessgym.com/)
    - **Papers**:
        - [Red Teaming Language Models to Reduce Harms (2022)](https://arxiv.org/abs/2202.03286)

---

## Module 14: Ethics & Security

**Objective:** Ensure responsible AI development by focusing on the ethical and security implications of LLMs.

- [ ] **Constitutional AI and Ethical Constraints**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explore Constitutional AI and methods for programming ethical constraints and guidelines into LLMs.
    - **Practical Tasks:**
        - Implement basic ethical constraint programming for an LLM using Constitutional AI principles.
    - **Additional Sources**:
        - [Constitutional AI by Anthropic](https://www.anthropic.com/constitutional-ai)
        - [AI Ethics Resources by the AI Ethics Lab](https://aiethicslab.com/resources/)
        - [Center for AI and Digital Policy](https://www.ai-policy.org/)
    - **Papers**:
        - [Constitutional AI: Harmlessness from AI Feedback (2022)](https://arxiv.org/abs/2212.08073)

- [ ] **Model Watermarking and Generation Traceability**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Learn about model watermarking techniques to ensure generation traceability and detect AI-generated content.
    - **Practical Tasks:**
        - Implement a basic watermarking technique for text generated by an LLM.
    - **Additional Sources**:
        - [Watermarking for AI-Generated Content](https://ai.google/responsibilities/our-approach-to-responsible-ai/watermarking/) (Google AI)
        - [Fairly Watermarked Dataset](https://fairly-watermarked-dataset.org/)
        - [Robust Watermarking of Large Language Models](https://arxiv.org/abs/2301.10226)
    - **Papers**:
        - [Robust Watermarking of Large Language Models (2023)](https://arxiv.org/abs/2301.10226)

- [ ] **Privacy Preservation in LLM Applications**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implement and understand privacy-preserving methods like differential privacy for LLM applications to protect user data.
    - **Practical Tasks:**
        - Explore and implement differential privacy techniques in an LLM application.
    - **Additional Sources**:
        - [Differential Privacy Explained](https://programmingdp.com/)
        - [PyTorch Privacy Library](https://pytorch.org/docs/stable/privacy.html)
        - [TensorFlow Privacy Library](https://www.tensorflow.org/privacy)
    - **Papers**:
        - [The Algorithmic Foundations of Differential Privacy (2014)](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) (Book on Differential Privacy)
        - [Privacy-preserving Machine Learning: Opportunities and Challenges (2017)](https://arxiv.org/abs/1702.07576)

---

## Module 15: Maintenance & Monitoring

**Objective:** Establish practices for the ongoing maintenance and monitoring of LLM deployments to ensure reliability and performance over time.

- [ ] **Drift Detection and Model Retraining Strategies**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Implement drift detection mechanisms to monitor for concept drift and trigger model retraining pipelines.
    - **Practical Tasks:**
        - Implement concept drift monitoring for an LLM application and set up automated retraining triggers.
    - **Additional Sources**:
        - [Concept Drift Detection Methods](https://riverml.xyz/latest/drift/) (River library for online ML)
        - [Evidently AI for Model Monitoring](https://evidentlyai.com/)
        - [Fiddler AI Model Monitoring Platform](https://www.fiddler.ai/)
    - **Papers**:
        - [Drift Detection in Data Streams (2004)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=10.1.1.65.2455) (Early work on drift detection)

- [ ] **Explainability Dashboard and Interpretability Tools**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Integrate explainability tools (SHAP, LIME) into a dashboard for monitoring and understanding LLM behavior in production.
    - **Practical Tasks:**
        - Integrate SHAP or LIME for explainability into a monitoring dashboard for an LLM.
    - **Additional Sources**:
        - [SHAP (SHapley Additive exPlanations) Library](https://shap.readthedocs.io/en/latest/)
        - [LIME (Local Interpretable Model-agnostic Explanations) Library](https://github.com/marcotcr/lime)
        - [InterpretML Toolkit by Microsoft](https://interpret.ml/)
    - **Papers**:
        - [Explaining Explanations: An Overview of Interpretability of Machine Learning (2018)](https://arxiv.org/abs/1806.00069)

- [ ] **Continuous Learning and Online Adaptation Pipelines**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Design and implement continuous learning pipelines for LLMs to enable online adaptation and improvement over time.
    - **Practical Tasks:**
        - Design an online adaptation pipeline for an LLM to continuously learn from new data.
    - **Additional Sources**:
        - [Online Machine Learning with River](https://riverml.xyz/latest/)
        - [Continual Learning in Neural Networks: A Survey (2019)](https://arxiv.org/abs/1909.08534)
        - [Lifelong Learning for NLP](https://nlplifelonglearning.org/)
    - **Papers**:
        - [Continual Learning in Neural Networks: A Survey (2019)](https://arxiv.org/abs/1909.08534)

---

## Module 16: Multimodal Systems

**Objective:** Focus on building multimodal systems that integrate LLMs with other modalities like images, audio, and video.

- [ ] **Image-to-Text Generation with CLIP and LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Build image captioning and image-to-text generation systems using CLIP for visual understanding and LLMs for text generation.
    - **Practical Tasks:**
        - Implement CLIP-guided image captioning using an LLM.
    - **Additional Sources**:
        - [Hugging Face Transformers for CLIP](https://huggingface.co/docs/transformers/model_doc/clip)
        - [Conceptual Captions Dataset](https://ai.google.com/research/ConceptualCaptions/)
        - [COCO Caption Dataset](https://cocodataset.org/#captions-challenge2015)
    - **Papers**:
        - [Learning Transferable Visual Models From Natural Language Supervision (CLIP) (2021)](https://arxiv.org/abs/2103.00020)

- [ ] **Audio Understanding and Integration with LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Integrate audio processing models like Whisper with LLMs for tasks like speech-to-text and audio-based question answering.
    - **Practical Tasks:**
        - Implement Whisper+LLM integration for audio understanding tasks.
    - **Additional Sources**:
        - [Whisper by OpenAI](https://openai.com/research/whisper)
        - [Hugging Face Transformers for Whisper](https://huggingface.co/docs/transformers/model_doc/whisper)
        - [LibriSpeech Dataset for Speech Recognition](https://www.openslr.org/12/)
    - **Papers**:
        - [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper) (2022)](https://arxiv.org/abs/2212.04356)

- [ ] **Video Summarization and Analysis with Multimodal LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Develop video summarization and analysis systems by combining frame-level visual information and transcript-based textual understanding using multimodal LLMs.
    - **Practical Tasks:**
        - Implement video summarization using frame and transcript analysis with multimodal LLMs.
    - **Additional Sources**:
        - [VideoMAE (Masked Autoencoders are Efficient Video Learners)](https://arxiv.org/abs/2203.12602)
        - [TimeSformer (Is Space-Time Attention All You Need for Video Understanding?)](https://arxiv.org/abs/2102.05095)
        - [YouTube-8M Dataset for Video Understanding](https://research.google.com/youtube8m/)
    - **Papers**:
        - [Video Summarization using Deep Neural Networks: A Survey (2019)](https://arxiv.org/abs/1907.07843) (Survey on Video Summarization)

---

## Module 17: Capstone Project

**Objective:**  Apply the knowledge and skills gained throughout the course to a comprehensive capstone project.

- [ ] **Full-Stack LLM Application Development**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Develop a full-stack application powered by LLMs, including custom fine-tuning, deployment, and monitoring.
    - **Practical Tasks:**
        - Build and deploy a complete full-stack application leveraging custom fine-tuning and monitoring for LLMs.

- [ ] **Research Paper Reproduction and Extension**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Choose a landmark research paper in the LLM field, reproduce its results, and extend it with novel ideas or experiments.
    - **Practical Tasks:**
        - Reimplement and extend a significant research paper in the field of Large Language Models.

- [ ] **Energy Efficiency and Carbon Footprint Study of LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Conduct a research study on the energy efficiency and carbon footprint of training and deploying LLMs, proposing methods for reducing environmental impact.
    - **Practical Tasks:**
        - Conduct a carbon footprint analysis of training and deploying specific LLM architectures.

---

## Module 18: Emerging Trends

**Objective:** Stay ahead of the curve by exploring emerging trends and future directions in LLM research and development.

- [ ] **Sparse Mixture-of-Experts (MoE) Models**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explore Sparse Mixture-of-Experts architectures for scaling LLMs efficiently, focusing on dynamic routing and sparsity.
    - **Practical Tasks:**
        - Implement a basic Sparse Mixture-of-Experts layer with dynamic routing.
    - **Additional Sources**:
        - [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
        - [Mixture-of-Experts Explained](https://huggingface.co/blog/moe)
        - [DeepSpeed MoE Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
    - **Papers**:
        - [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (2021)](https://arxiv.org/abs/2101.03961)

- [ ] **Quantum Machine Learning for LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Explore the potential of quantum machine learning algorithms for enhancing LLMs, focusing on quantum attention mechanisms and hybrid quantum-classical approaches.
    - **Practical Tasks:**
        - Research and explore the theoretical advantages of quantum attention mechanisms.
    - **Additional Sources**:
        - [Quantum Machine Learning Tutorials by Xanadu](https://pennylane.ai/qml/demonstrations/) (PennyLane library)
        - [Google Quantum AI](https://quantumai.google/)
        - [IBM Quantum Experience](https://quantum-computing.ibm.com/)
    - **Papers**:
        - [Quantum Machine Learning: What Can Quantum Computing Offer Machine Learning? (2017)](https://www.nature.com/articles/nature23479) (Survey on Quantum ML)
        - [Quantum Attention (2021)](https://arxiv.org/abs/2107.02089)

- [ ] **Neurological Modeling and Brain-Inspired LLMs**  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#)
    - **Description:** Investigate brain-inspired approaches to LLMs, exploring neurological modeling, fMRI-to-text decoding, and cognitive architectures.
    - **Practical Tasks:**
        - Research and explore techniques for fMRI-to-text decoding using LLMs.
    - **Additional Sources**:
        - [Human Brain Project](https://www.humanbrainproject.eu/en/)
        - [Allen Institute for Brain Science](https://alleninstitute.org/what-we-do/brain-science/)
        - [Brain-Score Benchmark for Brain-Like AI](https://brain-score.org/)
    - **Papers**:
        - [fMRI Decoding of Spoken Sentences Based on Word Embeddings (2017)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5478015/) (Example of fMRI decoding)
        - [Cognitive Architectures: Research Trends (2019)](https://link.springer.com/referenceworkentry/10.1007/978-3-319-47971-7_15-1) (Survey on Cognitive Architectures)
