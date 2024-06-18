# RAG Powered AI Assistant for Requirement Engineering

[![Language](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![Framework](https://img.shields.io/badge/-LangChain-1C3C3C?style=for-the-badge&logo=Langchain&logoColor=white)]()
[![Framework](https://img.shields.io/badge/-Docker-2496ED?style=for-the-badge&logo=Docker&logoColor=white)]()
[![Framework](https://img.shields.io/badge/-Milvus-00A1EA?style=for-the-badge&logo=Milvus&logoColor=white)]()
[![Framework](https://img.shields.io/badge/-Flask-000000?style=for-the-badge&logo=Flask&logoColor=white)]()
[![Framework](https://img.shields.io/badge/-React-61DAFB?style=for-the-badge&logo=React&logoColor=white)]()

---
## About The Project

Being one of the crucial steps during a software development process, Requirement engineering
(RE) plays a vital role in safety-critical systems. It is important as all the features of the system would be centered around critical requirements. Therefore, the requirements evolve through the software development lifecycle by means of resource-intensive, time-consuming, expert supervision. The RE tasks such as classification, tracing, ambiguity detection, prioritization, etc. have been explored with the help of Machine Learning and Deep Learning models. However, the majority of these modes are based on supervised learning. These supervised tasks require the participation of experts and practitioners and a huge amount of task-specific labeled data.

Large Language Models (LLMs) are Generative Artificial (GenAI) models are specialized in natural language-specific tasks and they can be highly useful when used in the Requirements Engineering domain. LLMs can be pre-trained to focus and generate specific tasks if the models receive accurate and concise prompts. These particular ways of designing prompts are referred to as Prompt Engineering. When following particular prompt patterns, this Prompt Engineering techniques can help requirement engineers in establishing specific RE-related downstream tasks. Moreover, LLMs follow the procedures of few-shot learning and can easily adapt to and use the knowledge given to it as a context in the form of a few examples.

Providing an LLM that is pre-trained with a new knowledge base which is task-specific will add to its capabilities. We can enable these task-specific capabilities by fine-tuning the language models to an additional knowledge base. Also with the use of prompt engineering techniques, one can effectively enable an LLM model to acquire a seemingly endless source of knowledge without the need for retraining the model from scratch.

One such prompt engineering technique is Retrieval-Augmented Generation (RAG) which enables LLMs to perform Knowledge-intensive natural language processing tasks by enriching their knowledge base without the need to change the models’ parameters. RAG combines an information retrieval component with a language generator model and generates knowledge-specific tasks that were, previously, absent in the language generation model’s knowledge base. This effectively opens the opportunity to deploy adaptive chat systems that can help experts and practitioners perform RE tasks more efficiently and quickly. As chat assistants can adapt and learn RE-specific tasks through RAG, the continuous integration of knowledge can virtually teach the assistants any kind of specialized workload.
