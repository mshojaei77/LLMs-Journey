# Module 15: Function Calling and AI Agents

### Function Calling Fundamentals
- **Description**: Learn how to enable LLMs to interact with external functions and APIs.
- **Concepts Covered**: `function calling`, `API integration`, `structured outputs`, `JSON schemas`, `database querying`, `parallel function calls`, `Pythonic function calling`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![OpenAI Function Calling Guide](https://badgen.net/badge/Docs/OpenAI_Function_Calling_Guide/green)](https://platform.openai.com/docs/guides/function-calling) | [![Querying Databases with Function Calling (2024)](https://badgen.net/badge/Paper/Querying_Databases_with_Function_Calling_(2024)/purple)](https://arxiv.org/pdf/2502.00032) |
| [![LangChain Function Calling](https://badgen.net/badge/Docs/LangChain_Function_Calling/green)](https://python.langchain.com/docs/modules/model_io/output_parsers/structured) | [![Dria Agent α Blog Post](https://badgen.net/badge/Blog/Dria_Agent_α_Blog_Post/pink)](https://huggingface.co/blog/andthattoo/dria-agent-a) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![OpenAI Function Calling API](https://badgen.net/badge/API Provider/OpenAI_Function_Calling_API/blue)](https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions) | [![Gorilla Database Query Tool](https://badgen.net/badge/Github Repository/Gorilla_Database_Query_Tool/cyan)](https://github.com/weaviate/gorilla) |
| [![LangChain](https://badgen.net/badge/Framework/LangChain/green)](https://python.langchain.com/) | [![Dria Agent Models](https://badgen.net/badge/Hugging Face Model/Dria_Agent_Models/yellow)](https://huggingface.co/driaforall) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Function Calling Basics](https://badgen.net/badge/Notebook/Function%20Calling%20Basics/orange)](notebooks/function_calling_basics.ipynb) | Implementing basic function calling with OpenAI API |
| [![Advanced Function Calls](https://badgen.net/badge/Notebook/Advanced%20Function%20Calls/orange)](notebooks/advanced_function_calls.ipynb) | Working with parallel function calls and JSON schemas |
| [![Database Integration](https://badgen.net/badge/Notebook/Database%20Integration/orange)](notebooks/database_integration.ipynb) | Using function calling for database queries |

### AI Agents & Autonomous Systems
- **Description**: Build autonomous AI agents and multi-agent systems that can plan and execute complex tasks.
- **Concepts Covered**: `autonomous agents`, `planning`, `task decomposition`, `tool use`, `multi-agent systems`, `agent communication`, `collaboration protocols`, `emergent behavior`, `layered memory`, `orchestration`, `multi-step workflows`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Building AI Agents with LangChain](https://badgen.net/badge/Docs/Building_AI_Agents_with_LangChain/green)](https://python.langchain.com/docs/modules/agents/) | [![Chain of Agents: LLMs Collaborating on Long-Context Tasks](https://badgen.net/badge/Paper/Chain_of_Agents:_LLMs_Collaborating_on_Long-Context_Tasks/purple)](https://research.google/chain-of-agents) |
| [![AutoGPT Documentation](https://badgen.net/badge/Docs/AutoGPT_Documentation/green)](https://docs.agpt.co/) | [![Multi-Agent Systems Overview](https://badgen.net/badge/Paper/Multi-Agent_Systems_Overview/purple)](https://arxiv.org/abs/2306.15330) |
| [![BabyAGI Paper](https://badgen.net/badge/Paper/BabyAGI_Paper/purple)](https://arxiv.org/abs/2305.12366) | [![Building AI Agents Newsletter](https://badgen.net/badge/Blog/Building_AI_Agents_Newsletter/pink)](https://buildingaiagents.substack.com/) |
| [![Anthropic: Building Effective Agents](https://badgen.net/badge/Blog/Anthropic:_Building_Effective_Agents/pink)](https://www.anthropic.com/research/building-effective-agents) | [![Hugging Face Agents Course](https://badgen.net/badge/Hugging Face Model/Hugging_Face_Agents_Course/yellow)](https://huggingface.co/agents-course) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![LangChain Agents](https://badgen.net/badge/Framework/LangChain_Agents/green)](https://python.langchain.com/docs/modules/agents/) | [![Langflow](https://badgen.net/badge/Github Repository/Langflow/cyan)](https://github.com/logspace-ai/langflow) |
| [![AutoGen](https://badgen.net/badge/Github Repository/AutoGen/cyan)](https://github.com/microsoft/autogen) | [![N8N](https://badgen.net/badge/Website/N8N/blue)](https://n8n.io/) |
| [![CrewAI](https://badgen.net/badge/Github Repository/CrewAI/cyan)](https://github.com/joaomdmoura/crewAI) | [![ComfyUI](https://badgen.net/badge/Github Repository/ComfyUI/cyan)](https://github.com/comfyanonymous/ComfyUI) |
| [![MetaGPT](https://badgen.net/badge/Github Repository/MetaGPT/cyan)](https://github.com/geekan/MetaGPT) | [![Smyth OS](https://badgen.net/badge/Website/Smyth_OS/blue)](https://smythos.com/) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Basic Agent Development](https://badgen.net/badge/Notebook/Basic%20Agent%20Development/orange)](notebooks/basic_agent_dev.ipynb) | Creating simple autonomous agents with LangChain |
| [![Multi-Agent Systems](https://badgen.net/badge/Notebook/Multi-Agent%20Systems/orange)](notebooks/multi_agent_systems.ipynb) | Building collaborative agent systems |
| [![Complex Workflows](https://badgen.net/badge/Notebook/Complex%20Workflows/orange)](notebooks/complex_workflows.ipynb) | Implementing multi-step agent workflows |

### Agent Evaluation & External Tools Integration
- **Description**: Implement evaluation frameworks and integrate external tools to enhance agent capabilities.
- **Concepts Covered**: `LLM-as-judge`, `quality metrics`, `evaluation frameworks`, `API integration`, `tool libraries`, `web services`, `data sources`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![LangSmith Documentation](https://badgen.net/badge/Docs/LangSmith_Documentation/green)](https://docs.smith.langchain.com/) | [![Building Custom Tools Guide](https://badgen.net/badge/Docs/Building_Custom_Tools_Guide/green)](https://python.langchain.com/docs/modules/agents/tools/custom_tools) |
| [![LangChain Tools Documentation](https://badgen.net/badge/Docs/LangChain_Tools_Documentation/green)](https://python.langchain.com/docs/integrations/tools) | [![Hugging Face Cookbook: Evaluating AI Search Engines](https://badgen.net/badge/Hugging Face Model/Hugging_Face_Cookbook:_Evaluating_AI_Search_Engines/yellow)](https://huggingface.co/learn/cookbook/llm_judge_evaluating_ai_search_engines_with_judges_library) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![LangSmith](https://badgen.net/badge/Framework/LangSmith/green)](https://smith.langchain.com/) | [![Perplexity AI](https://badgen.net/badge/Website/Perplexity_AI/blue)](https://www.perplexity.ai/) |
| [![OpenAI Evals](https://badgen.net/badge/Github Repository/OpenAI_Evals/cyan)](https://github.com/openai/evals) | [![Exa AI](https://badgen.net/badge/Website/Exa_AI/blue)](https://exa.ai/) |
| [![SerpAPI](https://badgen.net/badge/API Provider/SerpAPI/blue)](https://serpapi.com/) | [![Google Gemini](https://badgen.net/badge/Hugging Face Model/Google_Gemini/yellow)](https://deepmind.google/technologies/gemini/) |
| [![GitHub API](https://badgen.net/badge/API Provider/GitHub_API/blue)](https://docs.github.com/en/rest) | [![Beautiful Soup](https://badgen.net/badge/Docs/Beautiful_Soup/green)](https://www.crummy.com/software/BeautifulSoup/) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Evaluation Setup](https://badgen.net/badge/Notebook/Evaluation%20Setup/orange)](notebooks/evaluation_setup.ipynb) | Setting up evaluation frameworks with LangSmith |
| [![Tool Integration](https://badgen.net/badge/Notebook/Tool%20Integration/orange)](notebooks/tool_integration.ipynb) | Integrating external APIs and services |
| [![Custom Tools](https://badgen.net/badge/Notebook/Custom%20Tools/orange)](notebooks/custom_tools.ipynb) | Building and deploying custom agent tools |

### AI Agents in Different Domains
- **Description**: Build AI agents for different domains, including medical, finance, and education.
- **Concepts Covered**: `medical`, `finance`, `education`, `agentic reasoning`, `multimodal integration`, `expert systems`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![MedRAX Paper](https://badgen.net/badge/Paper/MedRAX_Paper/purple)](https://arxiv.org/abs/2502.02673) | [![AI Hedge Fund Implementation](https://badgen.net/badge/Github Repository/AI_Hedge_Fund_Implementation/cyan)](https://github.com/virattt/ai-hedge-fund) |
| [![TradingGPT: Multi-Agent System with Layered Memory](https://badgen.net/badge/Paper/TradingGPT:_Multi-Agent_System_with_Layered_Memory/purple)](https://arxiv.org/pdf/2309.03736) | [![ChestAgentBench Dataset](https://badgen.net/badge/Hugging Face Dataset/ChestAgentBench_Dataset/yellow)](https://huggingface.co/datasets/wanglab/chest-agent-bench) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Social Analyzer](https://badgen.net/badge/Github Repository/Social%20Analyzer/cyan)](https://github.com/qeeqbox/social-analyzer) | [![Intelligence X](https://badgen.net/badge/Tool/Intelligence%20X/blue)](https://intelx.io/) |
| [![Bright Data](https://badgen.net/badge/API Provider/Bright%20Data/blue)](https://brightdata.com/) | [![OSINT Combine](https://badgen.net/badge/Tool/OSINT%20Combine/blue)](https://www.osintcombine.com/) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Medical Agents](https://badgen.net/badge/Notebook/Medical%20Agents/orange)](notebooks/medical_agents.ipynb) | Building diagnostic and analysis agents |
| [![Financial Agents](https://badgen.net/badge/Notebook/Financial%20Agents/orange)](notebooks/financial_agents.ipynb) | Implementing trading and analysis agents |
| [![Educational Agents](https://badgen.net/badge/Notebook/Educational%20Agents/orange)](notebooks/educational_agents.ipynb) | Creating tutoring and assessment agents |

### External Data Sources for AI Agents
- **Description**: Integrate diverse external data sources to enhance AI agent capabilities with real-time and historical information.
- **Concepts Covered**: `social media data`, `OSINT integration`, `data aggregation`, `cross-platform analysis`, `real-time monitoring`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![OSINT Framework](https://badgen.net/badge/Docs/OSINT%20Framework/green)](https://osintframework.com/) | [![Social Media Intelligence Guide](https://badgen.net/badge/Docs/Social%20Media%20Guide/green)](https://www.bellingcat.com/resources/how-tos/2019/12/10/social-media-intelligence-guide/) |
| [![OSINT Techniques](https://badgen.net/badge/Docs/OSINT%20Techniques/green)](https://www.osinttechniques.com/) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![TweetDeck](https://badgen.net/badge/Tool/TweetDeck/blue)](https://tweetdeck.twitter.com/) | [![Sherlock](https://badgen.net/badge/Github Repository/Sherlock/cyan)](https://github.com/sherlock-project/sherlock) |
| [![Osintgram](https://badgen.net/badge/Github Repository/Osintgram/cyan)](https://github.com/Datalux/Osintgram) | [![Alfred OSINT](https://badgen.net/badge/Github Repository/Alfred%20OSINT/cyan)](https://github.com/Alfredredbird/alfred) |
| [![RocketReach](https://badgen.net/badge/Tool/RocketReach/blue)](https://rocketreach.co/) | [![Social Searcher](https://badgen.net/badge/Tool/Social%20Searcher/blue)](https://www.social-searcher.com/) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Social Media Integration](https://badgen.net/badge/Notebook/Social%20Media%20Integration/orange)](notebooks/social_media_integration.ipynb) | Connecting agents to social platforms |
| [![OSINT Collection](https://badgen.net/badge/Notebook/OSINT%20Collection/orange)](notebooks/osint_collection.ipynb) | Building data collection pipelines |
| [![Cross-Platform Analysis](https://badgen.net/badge/Notebook/Cross-Platform%20Analysis/orange)](notebooks/cross_platform_analysis.ipynb) | Implementing multi-source data analysis |



### Intelligent Agents & Tool Integration
- **Description**: Build agents that integrate LLMs with external tools and APIs for automation.
- **Concepts Covered**: `intelligent agents`, `automation`, `tool integration`, `API interaction`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![LangChain Agents Guide](https://badgen.net/badge/Docs/LangChain_Agents_Guide/green)](https://python.langchain.com/docs/modules/agents/) | [![Agent-Based Modeling](https://badgen.net/badge/Website/Agent-Based_Modeling/blue)](https://www.jasss.org/16/2/5.html) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![LangChain](https://badgen.net/badge/Github%20Repository/LangChain/cyan)](https://github.com/hwchase17/langchain) | [![AutoGPT](https://badgen.net/badge/Github%20Repository/AutoGPT/cyan)](https://github.com/Significant-Gravitas/Auto-GPT) |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Basic Agent](https://badgen.net/badge/Notebook/Basic%20Agent/orange)](notebooks/basic_agent.ipynb) | Creating a simple LLM-powered agent |
| [![Tool Integration](https://badgen.net/badge/Notebook/Tool%20Integration/orange)](notebooks/tool_integration.ipynb) | Connecting agents with external tools |
| [![Complex Workflows](https://badgen.net/badge/Notebook/Complex%20Workflows/orange)](notebooks/complex_workflows.ipynb) | Building multi-step agent workflows |

### Custom LLM Applications
- **Description**: Develop tailored LLM solutions for specific business or research needs.
- **Concepts Covered**: `custom applications`, `domain adaptation`, `specialized models`, `AI agents`, `RAG implementations`, `scalable solutions`

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Building Custom LLMs](https://badgen.net/badge/Tutorial/Building_Custom_LLMs/blue)](https://www.deeplearning.ai/short-courses/building-applications-with-vector-databases/) | [![Domain-Specific Language Models](https://badgen.net/badge/Paper/Domain-Specific_Language_Models/purple)](https://arxiv.org/abs/2004.06547) |
| [![Reflex LLM Examples](https://badgen.net/badge/Github%20Repository/Reflex_LLM_Examples/cyan)](https://github.com/reflex-dev/reflex-llm-examples) | |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![Hugging Face Transformers](https://badgen.net/badge/Hugging%20Face%20Model/Hugging_Face_Transformers/yellow)](https://huggingface.co/) | [![OpenSesame](https://badgen.net/badge/Website/OpenSesame/blue)](https://opensesame.dev/) |
| [![Custom Datasets](https://badgen.net/badge/Hugging%20Face%20Dataset/Custom_Datasets/yellow)](https://huggingface.co/docs/datasets/loading) | [![Readwise](https://badgen.net/badge/Website/Readwise/blue)](https://readwise.io/) |
| [![Reflex](https://badgen.net/badge/Github%20Repository/Reflex/cyan)](https://github.com/reflex-dev/reflex) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![Domain Adaptation](https://badgen.net/badge/Notebook/Domain%20Adaptation/orange)](notebooks/domain_adaptation.ipynb) | Fine-tuning LLMs for specific domains |
| [![Custom Application](https://badgen.net/badge/Notebook/Custom%20Application/orange)](notebooks/custom_application.ipynb) | Building a specialized LLM application |
| [![RAG Implementation](https://badgen.net/badge/Notebook/RAG%20Implementation/orange)](notebooks/rag_implementation.ipynb) | Implementing retrieval-augmented generation |

### Document Processing and Structured Data Extraction

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![PDF Q&A with DeepSeek Tutorial](https://badgen.net/badge/Tutorial/PDF_Q&A_with_DeepSeek_Tutorial/blue)](https://youtube.com/watch?v=M6vZ6b75p9k&list=PLp01ObP3udmq2quR-RfrX4zNut_t_kNot) | [![Gemini PDF to Data Tutorial](https://badgen.net/badge/Tutorial/Gemini_PDF_to_Data_Tutorial/blue)](https://www.philschmid.de/gemini-pdf-to-data) |
| [![Gemini 2.0 File API Documentation](https://badgen.net/badge/Docs/Gemini_2.0_File_API_Documentation/green)](https://ai.google.dev/docs/file_api) | [![Pydantic Documentation](https://badgen.net/badge/Docs/Pydantic_Documentation/green)](https://docs.pydantic.dev/) |

#### Tools & Frameworks
| Core | Additional |
|-----------|----------|
| [![PDF Dino](https://badgen.net/badge/Website/PDF_Dino/blue)](https://pdfdino.com) | [![Parsr](https://badgen.net/badge/Github%20Repository/Parsr/cyan)](https://github.com/axa-group/Parsr) |
| [![Google Generative AI SDK](https://badgen.net/badge/Github%20Repository/Google_Generative_AI_SDK/cyan)](https://github.com/google/generative-ai-python) | [![PyPDF2](https://badgen.net/badge/Github%20Repository/PyPDF2/cyan)](https://pypdf2.readthedocs.io/) |
| [![Pydantic](https://badgen.net/badge/Github%20Repository/Pydantic/cyan)](https://github.com/pydantic/pydantic) | |

#### Guided Practice
| Notebook | Description |
|----------|-------------|
| [![PDF Processing](https://badgen.net/badge/Notebook/PDF%20Processing/orange)](notebooks/pdf_processing.ipynb) | Setting up PDF extraction pipeline |
| [![Data Structuring](https://badgen.net/badge/Notebook/Data%20Structuring/orange)](notebooks/data_structuring.ipynb) | Implementing structured data extraction |
| [![Token Management](https://badgen.net/badge/Notebook/Token%20Management/orange)](notebooks/token_management.ipynb) | Managing API tokens and file sizes |

### AI Research Assistant Development

#### Learning Sources
| Essential | Optional |
|-----------|----------|
| [![Deep Research Agent Implementation](https://badgen.net/badge/Github%20Repository/Deep_Research_Agent_Implementation/cyan)](https://github.com/dzhng/deep-research) | |