# Module 4: Transformer Architecture Deep Dive

### The Attention Mechanism
- **Description**: Discover how attention enables models to focus on relevant parts of the input.
- **Concepts Covered**: `attention`, `softmax`, `context vectors`
- **Learning Resources**:
  - [Transformers from Scratch](https://brandonrohrer.com/transformers) - Comprehensive guide covering core concepts and implementation details
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [Attention? Attention! – Lilian Weng](https://lilianweng.github.io/posts/2018-06-24-attention/)
- **Tools**:
  - [Hugging Face Transformers](https://huggingface.co/docs/transformers)
  - [BertViz](https://github.com/jessevig/bertviz)

### Self-Attention & Multi-Head Attention
- **Description**: Learn how self-attention allows tokens to weigh each other's importance and how multiple heads capture diverse relationships.
- **Concepts Covered**: `self-attention`, `multi-head attention`, `query-key-value`
- **Learning Resources**:
  - [Self-Attention Explained (Paper)](https://arxiv.org/abs/1706.03762)
  - [Multi-Head Attention Visualized](https://jalammar.github.io/illustrated-transformer/)
- **Tools**:

  - [TensorFlow](https://www.tensorflow.org/)

### Positional Encoding in Transformers
- **Description**: Add order information to token embeddings using positional encodings.
- **Concepts Covered**: `positional encoding`, `sinusoidal functions`, `learned embeddings`
- **Learning Resources**:
  - [Positional Encoding Explorer](https://github.com/jalammar/positional-encoding-explorer)
  - [Rotary Embeddings Guide](https://blog.eleuther.ai/rotary-embeddings/)
- **Tools**:

  - [TensorFlow](https://www.tensorflow.org/)

### Layer Normalization & Residual Connections
- **Description**: Improve training stability with normalization and skip connections.
- **Concepts Covered**: `layer normalization`, `residual connections`, `training stability`
- **Learning Resources**:
  - [Layer Normalization Deep Dive](https://leimao.github.io/blog/Layer-Normalization/)
  - [Residual Network Paper](https://arxiv.org/abs/1512.03385)
- **Tools**:
  - [PyTorch LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
  - [TensorFlow LayerNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)
