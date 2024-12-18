import os
import asyncio
import logging
from typing import List, Dict, Optional
import datetime

# External Libraries
import aiohttp
import async_timeout 
from dotenv import load_dotenv

# CrewAI and LLM Imports
from crewai import Agent, Task, Crew
from langchain_mistralai import ChatMistralAI

# Utility and Web Scraping Imports
from newspaper import Article
import ssl
import certifi
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebScraper:
    @staticmethod
    async def extract_article_content(url, timeout=10):
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
    def __init__(self, 
                 mistral_api_key=None, 
                 debug=True):
        """
        Initialize research system with Mistral API key
        """

        ## initialize Mistral keys
        self.mistral_api_key = mistral_api_key or os.getenv('MISTRAL_API_KEY')

        ## initialize LLM
        self.llm = ChatMistralAI(
            api_key=self.mistral_api_key,
            model="mistral-large-latest"
        )

        if debug:
            print("Mistral API key: ", bool(self.mistral_api_key))

    async def advanced_research_workflow(self, query, sources):
        """
        Simplified research workflow
        """
        try:
            extraction_tasks = [
                WebScraper.extract_article_content(source) 
                for source in sources
            ]
            extracted_contents = await asyncio.gather(*extraction_tasks)
            
            ## remove empty results
            valid_contents = [
                content for content in extracted_contents 
                if content is not None
            ]
            
            # Prepare research context
            research_context = "\n\n".join([
                f"Source: {content.get('title', 'Untitled')} ({content.get('url', 'No URL')})\n"
                f"Content: {content.get('text', '')[:1000]}..."
                for content in valid_contents
            ])
            
            # Use LLM directly for synthesis
            messages = [
                {"role": "system", "content": "You are a senior research analyst. Synthesize the following research context to answer the query precisely and comprehensively."},
                {"role": "user", "content": f"Query: {query}\n\nResearch Context:\n{research_context}"}
            ]
            
            response = self.llm.invoke(messages)
            
            return {
                'status': 'success',
                'query': query,
                'sources': sources,
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
    research_system = AdvancedResearchSystem()
    
    ## give links for scraping directly
    sources = [
        'https://www.theguardian.com/world/2024/dec/08/bashar-al-assad-has-fled-syria-but-where-is-the-former-dictator-now',
        'https://www.cnn.com/2024/12/16/middleeast/syria-bashar-assad-statement-intl/index.html'
    ]
    
    query = "Bashar al-Assad current location"
    results = await research_system.advanced_research_workflow(query, sources)
    
    print(results['research_summary'])

if __name__ == "__main__":
    asyncio.run(main())