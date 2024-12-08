import os
import asyncio
import logging
from typing import List, Dict, Optional

# External Libraries
import aiohttp
import async_timeout
from dotenv import load_dotenv

# CrewAI and LLM Imports
from crewai import Agent, Task, Crew, LLM
from langchain_mistralai import ChatMistralAI

# Search API Clients
import serpapi
import requests
from googleapiclient.discovery import build

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

class AdvancedResearchSystem:
    def __init__(self, 
                 mistral_api_key=None, 
                 google_api_key=None, 
                 google_cse_id=None,
                 serpapi_key=None, 
                 debug=True):
        """
        Initialize research system with multiple API keys
        """
        # API Keys with fallback to environment variables
        self.mistral_api_key = mistral_api_key or os.getenv('MISTRAL_API_KEY')
        self.google_api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = google_cse_id or os.getenv('GOOGLE_CSE_ID')
        self.serpapi_key = serpapi_key or os.getenv('SERPAPI_KEY')


        self.mistral_llm = LLM(
        api_key=self.mistral_api_key,
        model="mistral/mistral-large-latest",
        )

        if debug:
            print("Mistral API key: ", bool(self.mistral_api_key))
            print("Google API key: ", bool(self.google_api_key))
            print("Google CSE ID: ", bool(self.google_cse_id))
            print("SerpAPI key: ", bool(self.serpapi_key))

    class MultiSourceSearchEngine:
        def __init__(self, 
                     google_api_key=None, 
                     google_cse_id=None,
                     serpapi_key=None):
            self.google_api_key = google_api_key
            self.google_cse_id = google_cse_id
            self.serpapi_key = serpapi_key

        async def serpapi_search(self, query, num_results=5):
            """
            Perform search using SerpAPI
            """
            if not self.serpapi_key:
                logger.warning("SerpAPI key not configured")
                return []

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
                
                return [
                    {
                        'title': result.get('title', ''),
                        'link': result.get('link', ''),
                        'snippet': result.get('snippet', ''),
                        'source': 'SerpAPI'
                    }
                    for result in data.get('organic_results', [])
                ]
            except Exception as e:
                logger.error(f"SerpAPI Search error: {e}")
                return []
        async def google_search(self, query, num_results=5):


        async def comprehensive_search(self, query, num_results=5):
            """
            Perform comprehensive search 
            """
            try:
                search_results = await self.serpapi_search(query, num_results)
                return search_results
            except Exception as e:
                logger.error(f"Comprehensive search error: {e}")
                return []

    class AdvancedWebScraper:
        """
        Enhanced web scraper with content extraction
        """
        @staticmethod
        async def extract_article_content(url, timeout=10):
            """
            Extract article content with enhanced extraction capabilities
            """
            try:
                # Add SSL context to handle potential certificate issues
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    async with async_timeout.timeout(timeout):
                        async with session.get(url, headers=headers, ssl=ssl_context) as response:
                            if response.status == 200:
                                html = await response.text()
                                
                                # Use newspaper3k for extraction
                                article = Article(url)
                                article.set_html(html)
                                article.parse()
                                
                                return {
                                    'url': url,
                                    'title': article.title or urlparse(url).netloc,
                                    'text': article.text or 'No content extracted',
                                    'authors': article.authors or ['Unknown'],
                                    'publish_date': str(article.publish_date or 'Unknown'),
                                    'top_image': article.top_image,
                                    'keywords': article.keywords or []
                                }
                            else:
                                logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                                return None
            except Exception as e:
                logger.error(f"Content extraction error for {url}: {e}")
                return None

    async def advanced_research_workflow(self, query, num_sources=5):
        """
        Advanced research workflow with CrewAI integration
        """
        # Initialize search engine
        search_engine = self.MultiSourceSearchEngine(
            google_api_key=self.google_api_key,
            google_cse_id=self.google_cse_id,
            serpapi_key=self.serpapi_key
        )
        
        # Perform comprehensive search
        search_results = await search_engine.comprehensive_search(query, num_sources)
        
        # Extract content from search results
        content_tasks = [
            self.AdvancedWebScraper.extract_article_content(result['link']) 
            for result in search_results
        ]
        
        # Extract content concurrently
        extracted_contents = await asyncio.gather(*content_tasks)
        
        # Filter out None results
        valid_contents = [
            content for content in extracted_contents 
            if content is not None
        ]

        # Create CrewAI agents for research synthesis
        researcher = Agent(
            role='Senior Research Analyst',
            goal='Synthesize and extract key insights from research materials',
            backstory='An expert researcher with deep analytical skills and the ability to distill complex information',
            verbose=True,
            llm=self.mistral_llm
        )

        synthesizer = Agent(
            role='Research Synthesizer',
            goal='Create a comprehensive and coherent summary of research findings',
            backstory='A skilled writer who can transform raw research findings into a clear and coherent response',
            verbose=True,
            llm=self.mistral_llm
        )

        # Prepare research context
        research_context = "\n\n".join([
            f"Source: {content.get('title', 'Untitled')} ({content.get('url', 'No URL')})\n"
            f"Authors: {', '.join(content.get('authors', ['Unknown']))}\n"
            f"Keywords: {', '.join(content.get('keywords', []))}\n"
            f"Content: {content.get('text', '')[:1000]}"
            for content in valid_contents
        ])

        # Create tasks for CrewAI
        research_task = Task(
            description=f"""Analyze the research materials on the topic: '{query}'
            Extract insights, recent developments and other critical developments.
            Identify trends, and unique perspectives.
            Prepare a detailed analysis that highlights the most important findings. Stick to only what has been found""",
            agent=researcher,
            expected_output="A comprehensive analysis with key insights and main points"
        )

        synthesis_task = Task(
            description="""Based on the research analysis, create a cohesive and 
            well-structured summary that presents the findings in a clear, 
            engaging, and informative manner. Ensure the summary is concise 
            yet captures the depth of the research while staying true to the findings.""",
            agent=synthesizer,
            expected_output="A polished, synthesized summary of the research findings"
        )

        # Create and execute the crew
        crew = Crew(
            agents=[researcher, synthesizer],
            tasks=[research_task, synthesis_task],
            verbose=True  
        )

        # Kickoff the research process
        result = crew.kickoff(inputs={'research_context': research_context})

        return {
            'search_results': search_results,
            'extracted_contents': valid_contents,
            'research_context': research_context,
            'final_research_summary': result
        }

# Async main function for demonstration
async def main():
    # Initialize advanced research system
    research_system = AdvancedResearchSystem()
    
    # Conduct research
    query = "Microsoft Stock Price Current"
    results = await research_system.advanced_research_workflow(query)
    
    # Display results
    print("\nFinal Research Summary:")
    print(results['final_research_summary'])

# Run the research workflow
if __name__ == "__main__":
    asyncio.run(main())