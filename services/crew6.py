import os
import asyncio
import logging
from typing import List, Dict, Optional
import datetime

# External Libraries
import aiohttp
import async_timeout
from dotenv import load_dotenv
import requests

# Utility and Web Scraping Imports
from newspaper import Article
import ssl
import certifi
from urllib.parse import urlparse

# LLM Import
from langchain_mistralai import ChatMistralAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class SearchEngine:
    """
    Search engine using SerpAPI for dynamic source retrieval
    Uses SerpApi to search the internet
    """
    def __init__(self, serpapi_key=None):
        """
        Initialize search engine with SerpApi key
        """
        self.serpapi_key = serpapi_key or os.getenv('SERPAPI_KEY')
        
        if not self.serpapi_key:
            raise ValueError("SerpApi key is required. Check .env file.")

    def search(self, query, num_results=5):
        """
        Perform a search using SerpApi
        
        Args:
            query (str): Search query
            num_results (int, optional): Number of results to retrieve. Defaults to 5.
        
        Returns:
            List[Dict]: List of search results with title, link, and snippet
        """
        try:
            params = {
                'engine': 'google',
                'q': query,
                'api_key': self.serpapi_key,
                'num': num_results,
                'device': 'desktop',
                'lr': 'lang_en'
            }
            
            response = requests.get('https://serpapi.com/search', params=params)
            data = response.json()
            
            # Extract organic results
            results = data.get('organic_results', [])
            
            # Transform results to a consistent format
            processed_results = [
                {
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('snippet', '')
                }
                for result in results
            ]
            
            return processed_results
        
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return []

class WebScraper:
    """
    Web content extraction
    """
    @staticmethod
    async def extract_article_content(url, timeout=10):
        """
        Asynchronously extract content from a given URL
        
        Args:
            url (str): URL to extract content from
            timeout (int, optional): Timeout for extraction. Defaults to 10.
        
        Returns:
            Dict: Extracted article content or None
        """
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Accept':'*/*',
                    'Accept-Encoding':'gzip, deflate, br',
                }
                
                async with async_timeout.timeout(timeout):
                    async with session.get(url, headers=headers, ssl=ssl_context) as response:
                        if response.status == 200:
                            html = await response.text()
                            
                            article = Article(url)
                            article.set_html(html)
                            article.parse()
                            logger.info(f"Extracted content from {url} and size is {len(article.text)}")
                            return {
                                'url': url,
                                'title': article.title or urlparse(url).netloc,
                                'text': article.text or 'No content extracted',
                                'publish_date': str(article.publish_date or 'Unknown'),
                                'top_image': article.top_image,
                            }
                        else:
                            logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                            return None
        except Exception as e:
            logger.error(f"Content extraction error for {url}: {e}")
            return None

class AdvancedResearchSystem:
    """
    Comprehensive research system with search and content extraction
    """
    def __init__(self, 
                 mistral_api_key=None, 
                 serpapi_key=None,
                 debug=True):
        """
        Initialize research system with API keys
        
        Args:
            mistral_api_key (str, optional): Mistral API key
            serpapi_key (str, optional): SerpAPI key
            debug (bool, optional): Enable debug logging. Defaults to True.
        """
        # Initialize API keys
        self.mistral_api_key = mistral_api_key or os.getenv('MISTRAL_API_KEY')
        
        # Initialize search engine
        self.search_engine = SearchEngine(serpapi_key)
        
        # Initialize LLM
        self.llm = ChatMistralAI(
            api_key=self.mistral_api_key,
            model="mistral-large-latest"
        )

        if debug:
            print("Mistral API key: ", bool(self.mistral_api_key))
            print("SerpAPI key: ", bool(self.search_engine.serpapi_key))

    async def advanced_research_workflow(self, query, num_sources=5):
        """
        Comprehensive research workflow with dynamic source retrieval
        
        Args:
            query (str): Research query
            num_sources (int, optional): Number of sources to retrieve. Defaults to 5.
        
        Returns:
            Dict: Research results including query, sources, and summary
        """
        try:
            # 1. Search for relevant sources
            search_results = self.search_engine.search(query, num_sources)
            
            if not search_results:
                return {
                    'status': 'no_results',
                    'query': query,
                    'message': 'No sources found for the given query.'
                }
            
            # 2. Extract source URLs
            source_urls = [result['link'] for result in search_results]
            
            # 3. Content extraction
            extraction_tasks = [
                WebScraper.extract_article_content(url) 
                for url in source_urls
            ]
            extracted_contents = await asyncio.gather(*extraction_tasks)
            
            # 4. Filter valid contents
            valid_contents = [
                content for content in extracted_contents 
                if content is not None
            ]
            
            # 5. Prepare research context
            research_context = "\n\n".join([
                f"Source: {content.get('title', 'Untitled')} ({content.get('url', 'No URL')})\n"
                f"Content: {content.get('text', '')[:1000]}..."
                for content in valid_contents
            ])
            
            # 6. Use LLM for synthesis
            messages = [
                {"role": "system", "content": "You are a senior research analyst. Synthesize the following research context to answer the query precisely and comprehensively. Give the reply as a coherent paragraph"},
                {"role": "user", "content": f"Query: {query}\n\nResearch Context:\n{research_context}"}
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                'status': 'success',
                'query': query,
                'search_results': search_results,
                'sources': [content['url'] for content in valid_contents],
                'extracted_contents': valid_contents,
                'research_summary': response.content
            }
        
        except Exception as e:
            logger.error(f"Research workflow error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'query': query
            }

async def main():
    """
    Main function to demonstrate research system usage
    """
    # Initialize research system
    research_system = AdvancedResearchSystem()
    
    # Example queries
    queries = [
        "Bashar al-Assad current location",
    ]
    
    # Perform research for each query
    for query in queries:
        print(f"\n--- Researching: {query} ---")
        results = await research_system.advanced_research_workflow(query)
        
        if results['status'] == 'success':
            print("\n--- Research Summary ---")
            print(results['research_summary'])
            
            print("\n--- Sources ---")
            for source in results['sources']:
                print(source)
        else:
            print(f"Research failed: {results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())