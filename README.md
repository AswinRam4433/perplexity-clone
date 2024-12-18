# Perplexity AI Clone

## Overview

This project is an advanced research assistant that leverages AI and web search technologies to provide evidence-backed, comprehensive responses to user queries. Inspired by Perplexity AI and [Arvind Srinivas' interview](https://www.youtube.com/watch?v=r1Bi10Xt0fc) , the application combines web search, content extraction, and AI-powered summarization to deliver accurate and informative real-time results which conventional LLMs like ChatGPT and Claude can't do.

## Features

- Intelligent Web Search
- Content Extraction from Multiple Sources
- AI-Powered Summarization
- Evidence-Backed Responses

## Technology Stack

- Python
- Mistral Large Language Model
- SerpAPI
- Async Web Scraping

## Prerequisites

### API Keys (Required)
- Mistral API Key
- SerpAPI Key

## Installation

1. Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_mistral_api_key
SERPAPI_KEY=your_serpapi_key
```

## Workflow

1. User submits a query
2. SerpAPI searches the internet for relevant sources
3. Content is extracted from top sources
4. Mistral AI analyzes and summarizes the content
5. Response is generated with evidence and insights

## Limitations

- Depends on API rate limits. Uses exponential backoff to overcome this
- Quality of results varies with source availability
- Requires active internet connection

## Future Improvements

- Add local caching of search results
- Implement more sophisticated source validation
- Add user interface (CLI/Web)


## Disclaimer

This project has been developed using a variety of tools and techniques, including the use of AI-powered coding assistants like GitHub Copilot, ChatGPT, and Claude.
These tools were used to help generate code snippets and provide design suggestionss. While the final decisions and implementations were made by human developers, these tools were invaluable in accelerating the development workflow.
