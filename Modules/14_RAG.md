# Module 14: Prompt Engineering & RAG
### Prompt Engineering Techniques
- **Description**: Master the art of crafting effective prompts to guide LLM behavior.
- **Concepts Covered**: `prompt engineering`, `prompt design`, `few-shot learning`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Prompt Engineering Guide](https://badgen.net/badge/Docs/Prompt%20Engineering%20Guide/blue)](https://www.promptingguide.ai/) | |
| [![Best Practices for Prompt Engineering](https://badgen.net/badge/Docs/Best%20Practices/green)](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![OpenAI Playground](https://badgen.net/badge/Website/OpenAI%20Playground/blue)](https://platform.openai.com/playground) | |
| [![Hugging Face Spaces](https://badgen.net/badge/Hugging%20Face%20Model/Hugging%20Face%20Spaces/yellow)](https://huggingface.co/spaces) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Basic Prompting](https://badgen.net/badge/Notebook/Basic%20Prompting/orange)](notebooks/basic_prompting.ipynb) | Introduction to prompt engineering fundamentals |
| [![Few-Shot Learning](https://badgen.net/badge/Notebook/Few-Shot%20Learning/orange)](notebooks/few_shot_learning.ipynb) | Implementing few-shot learning techniques |
| [![Advanced Patterns](https://badgen.net/badge/Notebook/Advanced%20Patterns/orange)](notebooks/advanced_patterns.ipynb) | Working with complex prompting patterns |

### Context Engineering & Control
- **Description**: Learn to manipulate context and control mechanisms for precise LLM outputs.
- **Concepts Covered**: `context engineering`, `control codes`, `conditional generation`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Controlling Text Generation](https://badgen.net/badge/Blog/Controlling%20Text%20Generation/pink)](https://huggingface.co/blog/how-to-generate) | |
| [![CTRL: A Conditional Transformer Language Model](https://badgen.net/badge/Paper/CTRL%20Model/purple)](https://arxiv.org/abs/1909.05858) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Transformers](https://badgen.net/badge/Framework/Transformers/green)](https://huggingface.co/docs/transformers) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Context Management](https://badgen.net/badge/Notebook/Context%20Management/orange)](notebooks/context_management.ipynb) | Managing and manipulating context windows |
| [![Control Codes](https://badgen.net/badge/Notebook/Control%20Codes/orange)](notebooks/control_codes.ipynb) | Working with control codes and flags |
| [![Conditional Generation](https://badgen.net/badge/Notebook/Conditional%20Generation/orange)](notebooks/conditional_generation.ipynb) | Implementing conditional text generation |
### Retrieval-Augmented Generation (RAG)
- **Description**: Combine LLMs with external knowledge retrieval for enhanced, factual responses.
- **Concepts Covered**: `RAG`, `retrieval`, `knowledge augmentation`, `vector databases`, `citation detection`, `span classification`, `real-time relevance scoring`, `source verification`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![RAG Paper](https://badgen.net/badge/Paper/RAG/purple)](https://arxiv.org/abs/2005.11401) | [![HNSW for Vector Search Tutorial](https://badgen.net/badge/Video/HNSW%20Tutorial/red)](https://www.youtube.com/watch?v=QvKMwLjdK-s) |
| [![LangChain RAG Tutorial](https://badgen.net/badge/Docs/LangChain%20RAG%20Tutorial/green)](https://python.langchain.com/docs/use_cases/question_answering/) | [![RAG vs Fine-tuning](https://badgen.net/badge/Paper/RAG%20vs%20Fine--tuning/purple)](https://arxiv.org/abs/2401.08406) |
| [![Build RAG with Milvus and Ollama](https://badgen.net/badge/Tutorial/Milvus%20%2B%20Ollama/blue)](https://milvus.io/docs/build_RAG_with_milvus_and_ollama.md#Build-RAG-with-Milvus-and-Ollama) | [![Top Down Design of RAG Systems](https://badgen.net/badge/Blog/Top%20Down%20RAG%20Design/pink)](https://medium.com/@manaranjanp/top-down-design-of-rag-systems-part-1-user-and-query-profiling-184651586854) |
| [![Advanced RAG Techniques E-book](https://badgen.net/badge/Docs/Advanced%20RAG%20Techniques/green)](https://weaviate.io/ebooks/advanced-rag-techniques) | [![Agentic RAG Tutorial](https://badgen.net/badge/Video/Agentic%20RAG%20Tutorial/red)](https://www.youtube.com/watch?v=2Fu_GgS-Q4s) |
| [![Local Citation Detection System](https://badgen.net/badge/Tutorial/Local%20Citation%20Detection/blue)](https://twitter.com/MaziyarPanahi/status/1750672543417962766) | [![Span Classification for Document Relevance](https://badgen.net/badge/Tutorial/Span%20Classification/blue)](https://twitter.com/MaziyarPanahi/status/1750672543417962766) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![FAISS](https://badgen.net/badge/Framework/FAISS/green)](https://github.com/facebookresearch/faiss) | [![Chipper](https://badgen.net/badge/Github%20Repository/Chipper/cyan)](https://github.com/TilmanGriesel/chipper) |
| [![Pinecone](https://badgen.net/badge/API%20Provider/Pinecone/blue)](https://www.pinecone.io/) | [![Phida](https://badgen.net/badge/Framework/Phida/green)](https://github.com/phidatahq/phida) |
| [![Weaviate](https://badgen.net/badge/API%20Provider/Weaviate/blue)](https://weaviate.io/) | [![Upstash](https://badgen.net/badge/Database/Upstash/blue)](http://upstash.com) |
| [![Milvus](https://badgen.net/badge/Database/Milvus/blue)](https://milvus.io/) | [![Qdrant](https://badgen.net/badge/Database/Qdrant/blue)](https://qdrant.tech/) |
| [![Ollama](https://badgen.net/badge/Framework/Ollama/green)](https://ollama.ai/) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Basic RAG Pipeline](https://badgen.net/badge/Notebook/Basic%20RAG%20Pipeline/orange)](notebooks/basic_rag_pipeline.ipynb) | Implementing a simple RAG system with vector search |
| [![Citation Detection](https://badgen.net/badge/Notebook/Citation%20Detection/orange)](notebooks/citation_detection.ipynb) | Building a local citation detection system |
| [![Advanced RAG](https://badgen.net/badge/Notebook/Advanced%20RAG/orange)](notebooks/advanced_rag.ipynb) | Optimizing RAG with pre/post processing techniques |
| [![Real-time RAG](https://badgen.net/badge/Notebook/Real-time%20RAG/orange)](notebooks/realtime_rag.ipynb) | Implementing real-time data collection and processing |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![CoT Dataset Creation](https://badgen.net/badge/Notebook/CoT%20Dataset%20Creation/orange)](notebooks/cot_dataset_creation.ipynb) | Build chain-of-thought datasets from scratch |
| [![Data Quality Assessment](https://badgen.net/badge/Notebook/Data%20Quality%20Assessment/orange)](notebooks/data_quality_assessment.ipynb) | Implement filtering and verification techniques |
| [![Poetry Dataset Generation](https://badgen.net/badge/Notebook/Poetry%20Dataset%20Generation/orange)](notebooks/poetry_dataset_generation.ipynb) | Create specialized poetry training data |

