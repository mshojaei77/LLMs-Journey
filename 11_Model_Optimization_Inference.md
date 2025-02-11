# Module 11: Model Optimization for Inference

### Inference Speedup with KV-Cache

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [KV-Caching Explained](https://huggingface.co/docs/transformers/v4.29.1/en/perf_infer_gpu_fp16_accelerate) | [LMCache Documentation](https://docs.lmcache.ai/index.html) |
| [DeepSpeed Inference Tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/#kv-cache) | [vLLM Production Stack](https://github.com/vllm-project/production-stack) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [LMCache](https://docs.lmcache.ai/index.html) | [vLLM](https://github.com/vllm-project/vllm) |
| Kubernetes Integration | CacheBlend & CacheGen |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![KV-Cache Implementation](https://badgen.net/badge/Notebook/KV-Cache%20Implementation/orange)](notebooks/kv_cache_impl.ipynb) | Hands-on implementation of KV-caching for inference speedup |
| [![Shared Cache Setup](https://badgen.net/badge/Notebook/Shared%20Cache%20Setup/orange)](notebooks/shared_cache.ipynb) | Setting up shared KV cache for multiple LLM instances |
| [![Production Deployment](https://badgen.net/badge/Notebook/Production%20Deployment/orange)](notebooks/prod_deployment.ipynb) | Deploying with vLLM and Kubernetes integration |

### Quantization Techniques for Inference

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [GPTQ Paper](https://arxiv.org/abs/2210.17323) | [DeepSeek-R1 1.58-bit Dynamic Quantization](https://unsloth.ai/blog/deepseekr1-dynamic) |
| [AWQ Paper](https://arxiv.org/abs/2306.00978) | [Intel Neural Compressor](https://github.com/intel/neural-compressor) |
| [QLoRA Paper](https://arxiv.org/abs/2305.14314) | [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) |
| [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/main/quantization) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) | [ONNX](https://onnx.ai/) |
| [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| [ExLlama](https://github.com/turboderp/exllama) | |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Basic Quantization](https://badgen.net/badge/Notebook/Basic%20Quantization/orange)](notebooks/basic_quantization.ipynb) | Implementing 4/8-bit quantization with BitsAndBytes |
| [![GPTQ Training](https://badgen.net/badge/Notebook/GPTQ%20Training/orange)](notebooks/gptq_training.ipynb) | Training and deploying GPTQ quantized models |
| [![Advanced Optimization](https://badgen.net/badge/Notebook/Advanced%20Optimization/orange)](notebooks/advanced_optimization.ipynb) | Working with TensorRT and ONNX optimization |

### Model Pruning for Efficient Inference

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [SparseML Pruning Guide](https://sparseml.neuralmagic.com/) | |
| [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [SparseML](https://sparseml.neuralmagic.com/) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Basic Pruning](https://badgen.net/badge/Notebook/Basic%20Pruning/orange)](notebooks/basic_pruning.ipynb) | Introduction to model pruning techniques |
| [![Production Pruning](https://badgen.net/badge/Notebook/Production%20Pruning/orange)](notebooks/production_pruning.ipynb) | Implementing pruning in production environments |

### Model Formats & Quantization Standards

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) | [Converting Models to GGUF](https://github.com/ggerganov/llama.cpp/blob/master/convert.py) |
| [GGML Technical Documentation](https://github.com/ggerganov/ggml/tree/master/docs) | |
| [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | [LM Studio](https://lmstudio.ai/) |
| [ctransformers](https://github.com/marella/ctransformers) | [Ollama](https://ollama.ai/) |
| [transformers-to-gguf](https://huggingface.co/spaces/lmstudio/convert-hf-to-gguf) | [text-generation-webui](https://github.com/oobabooga/text-generation-webui) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Format Conversion](https://badgen.net/badge/Notebook/Format%20Conversion/orange)](notebooks/format_conversion.ipynb) | Converting models to GGUF format |
| [![Deployment Setup](https://badgen.net/badge/Notebook/Deployment%20Setup/orange)](notebooks/deployment_setup.ipynb) | Setting up deployment with GGUF models |
| [![Performance Testing](https://badgen.net/badge/Notebook/Performance%20Testing/orange)](notebooks/performance_testing.ipynb) | Testing and validating converted models |

### Advanced Inference Optimization
- **Description**: Explore advanced techniques for optimizing inference performance, including SIMD optimizations and GPU acceleration.
- **Concepts Covered**: `SIMD`, `GPU`, `inference`, `performance`, `optimization`, `WASM`, `low-level optimization`, `dot product functions`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![SIMD Optimization PR Discussion](https://badgen.net/badge/Github%20Repository/SIMD%20Optimization%20PR%20Discussion/cyan)](https://github.com/ggerganov/llama.cpp/pull/11453) | [![LLM-Generated SIMD Optimization Prompt](https://badgen.net/badge/none/LLM--Generated%20SIMD%20Optimization%20Prompt/lightgray)](https://gist.github.com/ngxson/307140d24d80748bd683b396ba13be07) |
| [![WASM SIMD Development Example](https://badgen.net/badge/Github%20Repository/WASM%20SIMD%20Development%20Example/cyan)](https://github.com/ngxson/ggml/tree/xsn/wasm_simd_wip) | [![WLlama Benchmark Implementation](https://badgen.net/badge/Github%20Repository/WLlama%20Benchmark%20Implementation/cyan)](https://github.com/ngxson/wllama/pull/151) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![llama.cpp](https://badgen.net/badge/Github%20Repository/llama.cpp/cyan)](https://github.com/ggerganov/llama.cpp) | [![GGML](https://badgen.net/badge/Github%20Repository/GGML/cyan)](https://github.com/ggerganov/ggml) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![SIMD Optimization](https://badgen.net/badge/Notebook/SIMD%20Optimization/orange)](notebooks/simd_optimization.ipynb) | Implementing SIMD instructions for WASM |
| [![Dot Product Functions](https://badgen.net/badge/Notebook/Dot%20Product%20Functions/orange)](notebooks/dot_product_functions.ipynb) | Optimizing qX_K_q8_K and qX_0_q8_0 functions |
| [![Performance Benchmarking](https://badgen.net/badge/Notebook/Performance%20Benchmarking/orange)](notebooks/performance_benchmarking.ipynb) | Testing and validating optimizations |

### Extending Context Length
- **Description**: Explore techniques for extending LLM context windows beyond their original training length.
- **Concepts Covered**: `context extension`, `position interpolation`, `rotary embeddings`, `NTK-aware scaling`, `YaRN`, `dynamic NTK`, `star attention`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Position Interpolation Paper](https://badgen.net/badge/Paper/Position%20Interpolation%20Paper/purple)](https://arxiv.org/abs/2306.15595) | [![YaRN Paper](https://badgen.net/badge/Paper/YaRN%20Paper/purple)](https://arxiv.org/abs/2309.00071) |
| [![Dynamic NTK Paper](https://badgen.net/badge/Paper/Dynamic%20NTK%20Paper/purple)](https://arxiv.org/abs/2403.00831) | [![LongLoRA Paper](https://badgen.net/badge/Paper/LongLoRA%20Paper/purple)](https://arxiv.org/abs/2401.02397) |
| [![Context Length Scaling Laws](https://badgen.net/badge/Paper/Context%20Length%20Scaling%20Laws/purple)](https://arxiv.org/abs/2402.16617) | [![STAR Attention Paper](https://badgen.net/badge/none/STAR%20Attention%20Paper/lightgray)](https://arxiv.org/pdf/2411.17116) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![ExLlamaV2](https://badgen.net/badge/Github%20Repository/ExLlamaV2/cyan)](https://github.com/turboderp/exllamav2) | [![YaRN Implementation](https://badgen.net/badge/Github%20Repository/YaRN%20Implementation/cyan)](https://github.com/jquesnelle/yarn) |
| [![LongLoRA Implementation](https://badgen.net/badge/Github%20Repository/LongLoRA%20Implementation/cyan)](https://github.com/dvlab-research/LongLoRA) | [![vLLM Extended Context](https://badgen.net/badge/Docs/vLLM%20Extended%20Context/green)](https://docs.vllm.ai/en/latest/models/rope.html) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Position Interpolation](https://badgen.net/badge/Notebook/Position%20Interpolation/orange)](notebooks/position_interpolation.ipynb) | Implementing basic position interpolation |
| [![RoPE Scaling](https://badgen.net/badge/Notebook/RoPE%20Scaling/orange)](notebooks/rope_scaling.ipynb) | Working with NTK-aware and YaRN scaling |
| [![Attention Patterns](https://badgen.net/badge/Notebook/Attention%20Patterns/orange)](notebooks/attention_patterns.ipynb) | Optimizing attention for long contexts |

### Qwen 2.5 1M Context Models
- **Description**: Explore the first open-source models with 1 million token context length.

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Qwen 2.5 1M Context Models](https://badgen.net/badge/Hugging%20Face%20Dataset/Qwen%202.5%201M%20Context%20Models/yellow)](https://huggingface.co/collections/Qwen/qwen25-1m-679325716327ec07860530ba) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Qwen Models](https://badgen.net/badge/Hugging%20Face%20Models/Qwen%20Models/yellow)](https://huggingface.co/Qwen) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Long Context Setup](https://badgen.net/badge/Notebook/Long%20Context%20Setup/orange)](notebooks/long_context_setup.ipynb) | Setting up Qwen 2.5 for long contexts |
| [![Memory Management](https://badgen.net/badge/Notebook/Memory%20Management/orange)](notebooks/memory_management.ipynb) | Managing memory with 1M context windows |
| [![Performance Testing](https://badgen.net/badge/Notebook/Performance%20Testing/orange)](notebooks/performance_testing.ipynb) | Evaluating long context performance |
