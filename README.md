# Fashion Search AI
## Overview
**Fashion Search AI** is an AI-powered fashion search engine that leverages generative search capabilities and AI models like GPT-3.5 to intelligently understand and respond to user queries. The system uses a dataset from the Myntra fashion product catalog to recommend fashion items based on users’ search terms and preferences. The project integrates an advanced embedding layer, search layer, and generation layer, combining traditional search mechanisms with AI-generated content for a superior user experience.

## Objective
The primary goal is to develop an AI-powered system that efficiently searches and generates responses for fashion-related queries. The system not only finds fashion products based on descriptions but also generates detailed and contextually relevant answers, enhancing the overall search experience.

## Dataset
- Myntra Fashion Product Dataset: Available on Kaggle [here](https://www.kaggle.com/datasets/djagatiya/myntra-fashion-product-dataset/data).
- The dataset contains product descriptions, metadata, and other relevant information, which are preprocessed and embedded for efficient search.
## Dataset Chunking Decision
- No Chunking Implemented: The dataset size and structure were deemed manageable without the need for chunking. The entire dataset is processed as is, leveraging its row/column organization.
## System Architecture
The project is divided into three distinct layers: **Embedding**, **Search**, and **Generation**, each playing a crucial role in delivering high-quality search results and user-centric responses.

**1. Embedding Layer**
- **Preprocessing:** Data cleaning to ensure quality and consistency. This includes handling missing data, text standardization, and formatting.
- **Embeddings:** Product descriptions are embedded using OpenAI’s text-embedding-ada-002 model, generating numerical representations for search and retrieval.
- **Storage:** Embeddings are stored in a ChromaDB collection for efficient retrieval during search operations.
  
**2. Search Layer**
- **Semantic Search:** User queries are converted into embeddings, and the search is performed across the embedded product descriptions.
- **Cache Mechanism:** Implemented to store previous search results, improving system responsiveness by reusing cached results.
- **Re-Ranking:** A cross-encoder model from HuggingFace is employed to re-rank search results for better relevance and accuracy.
  
**3. Generation Layer**
- **Retrieval-Augmented Generation:** Uses GPT-3.5 to generate detailed, contextually relevant product recommendations. The system processes user queries along with refined search results, providing responses in natural language.
- **Final Prompt:** Carefully designed prompts ensure that the generated responses are user-friendly, summarizing key product features like brand, product name, and relevant details.
## Key Features
**1. AI-Powered Query Response:** Uses advanced AI models like GPT-3.5 to generate detailed answers to fashion-related queries.

**2. Semantic Search with Embeddings:** Efficient search using vector embeddings to identify fashion items that are contextually relevant to user preferences.

**3. Cross-Encoder Re-Ranking:** Improves search relevance by re-ranking retrieved results using a cross-encoder model.

**4. Cache for Faster Search:** Reduces search latency by caching previous results for reuse in similar queries.

**5. Context-Aware Responses:** The generation layer interprets user intent to provide actionable and personalized answers.
## Setup Instructions
**Step 1: Install Dependencies**
```
!pip install -U llama-index openai chromadb
```
**Step 2: Import Required Libraries**
```
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
import os
import openai
import pandas as pd
from sentence_transformers import CrossEncoder, util
```
**Step 3: Prepare Dataset**
- Download the Myntra dataset from Kaggle and preprocess the data using pandas for embedding generation.

**Step 4: Generate and Store Embeddings**
- Use text-embedding-ada-002 from OpenAI to create embeddings for the product descriptions and store them in a ChromaDB collection.
  
**Step 5: Implement Semantic Search**
- Query embeddings are compared with product embeddings in the database to retrieve top-matching fashion items. Cache results for efficiency.
  
**Step 6: Re-Rank Results**
- Use the cross-encoder model to re-rank search results, improving relevance to the user’s original query.
  
**Step 7: Generate Response Using GPT-3.5**
- Refined search results are passed to GPT-3.5, which generates a concise response, summarizing key product information in natural language.
  
## Challenges and Solutions
**1. Metadata Processing:** Initial issues with processing metadata were resolved through debugging and refining the code.

**2. Re-Ranking with Cross-Encoder:** Required testing different model combinations to find the most suitable cross-encoder for accurate re-ranking.
## Future Enhancements
**1. Web Application:** Develop a Flask-based web interface for better user interactivity.

**2. Enhanced Filters:** Add filters for more granular searches (e.g., price, material, brand).

**3. Improved Prompts:** Refine prompts for more accurate generation results.

**4. Chunking for Larger Datasets:** If the dataset grows, implement chunking to handle larger volumes more efficiently.
## Conclusion
The **Fashion Search AI** system successfully combines efficient search mechanisms with AI-generated responses to provide a seamless and personalized fashion search experience. The generative capabilities of GPT-3.5, combined with advanced search and re-ranking, significantly enhance the system’s ability to understand and fulfill user queries. Future development will focus on adding more interactive features, further improving the user experience.
