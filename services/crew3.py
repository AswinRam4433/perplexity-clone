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
            """
            Perform search using Google Custom Search API with enhanced features
            """
            if not self.google_api_key or not self.google_cse_id:
                logger.warning("Google API key or CSE ID not configured")
                return []

            try:
                service = build("customsearch", "v1", developerKey=self.google_api_key)
                
                # Enhanced search with additional parameters
                results = service.cse().list(
                    q=query,
                    cx=self.google_cse_id,
                    num=num_results,
                    fields="items(title,link,snippet)",
                    dateRestrict='d[1]'  # Results from last 24 hours
                ).execute()
                
                # Process and enrich search results
                processed_results = []
                for item in results.get('items', []):
                    result = {
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'source': 'Google Custom Search',
                        'timestamp': datetime.datetime.now().isoformat(),
                        'credibility_score': self._calculate_source_credibility(item.get('link', ''))
                    }
                    processed_results.append(result)
                
                return processed_results
            
            except Exception as e:
                logger.error(f"Google Search error: {e}")
                return []
        def _calculate_source_credibility(self, url):
            """
            Calculate a basic credibility score for a source
            """
            credible_domains = [
                'reuters.com', 'apnews.com', 'bloomberg.com', 
                'wsj.com', 'nytimes.com', 'bbc.com'
            ]
            
            for domain in credible_domains:
                if domain in url:
                    return 0.9  # High credibility
            
            return 0.5  # Default neutral credibility

        async def comprehensive_search(self, query, num_results=5):
            """
            Perform comprehensive search across multiple sources
            """
            try:
                # Parallelize search across multiple sources
                search_tasks = [
                    self.serpapi_search(query, num_results),
                    self.google_search(query, num_results)
                ]
                
                # Combine results, removing duplicates
                all_results = await asyncio.gather(*search_tasks)
                
                # Flatten results and remove duplicates based on links
                unique_results = []
                seen_links = set()
                for result_list in all_results:
                    for result in result_list:
                        if result['link'] not in seen_links:
                            unique_results.append(result)
                            seen_links.add(result['link'])
                
                return unique_results
            
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
                        'Accept':'*/*',
                        'Accept-Encoding':'gzip, deflate, br',


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
        Comprehensive asynchronous research workflow with multi-source search, 
        content extraction, and AI-powered synthesis.

        Args:
            query (str): The research query to investigate
            num_sources (int, optional): Number of sources to retrieve. Defaults to 5.

        Returns:
            dict: Comprehensive research results including search results, 
                extracted contents, and research summary
        """
    # Configure logging for the workflow
        logger = logging.getLogger(__name__)
        workflow_start_time = datetime.datetime.now()
        
        try:
            # 1. Initialize Search Engine
            search_engine = self.MultiSourceSearchEngine(
                google_api_key=self.google_api_key,
                google_cse_id=self.google_cse_id,
                serpapi_key=self.serpapi_key
            )
            
            # Log workflow initiation
            logger.info(f"Starting research workflow for query: '{query}'")
            
            # 2. Perform Comprehensive Search
            try:
                search_results = await search_engine.comprehensive_search(query, num_sources)
                
                if not search_results:
                    logger.warning(f"No search results found for query: '{query}'")
                    return {
                        'status': 'no_results',
                        'message': 'No search results could be retrieved',
                        'query': query
                    }
                
                # logger.info(f"Retrieved {len(search_results)} search results")
                logger.info(f"Retrieved search results")
            except Exception as search_error:
                logger.error(f"Search retrieval failed: {search_error}")
                return {
                    'status': 'search_error',
                    'message': str(search_error),
                    'query': query
                }
            
            # 3. Content Extraction
            async def safe_extract_content(result):
                """
                Safely extract content with timeout and error handling
                """
                try:
                    content = await asyncio.wait_for(
                        self.AdvancedWebScraper.extract_article_content(result['link']), 
                        timeout=15.0  # 15 seconds timeout for each extraction
                    )
                    return content
                except asyncio.TimeoutError:
                    logger.warning(f"Content extraction timed out for {result['link']}")
                    return None
                except Exception as extract_error:
                    logger.error(f"Content extraction failed for {result['link']}: {extract_error}")
                    return None
            
            # Concurrent content extraction
            extraction_tasks = [safe_extract_content(result) for result in search_results]
            extracted_contents = await asyncio.gather(*extraction_tasks)
            
            # Filter out None results
            valid_contents = [
                content for content in extracted_contents 
                if content is not None
            ]
            
            logger.info(f"Successfully extracted content from {len(valid_contents)} sources")
            
            # 4. Prepare Research Context
            research_context = "\n\n".join([
                f"Source: {content.get('title', 'Untitled')} ({content.get('url', 'No URL')})\n"
                f"Authors: {', '.join(content.get('authors', ['Unknown']))}\n"
                f"Keywords: {', '.join(content.get('keywords', []))}\n"
                f"Content Preview: {content.get('text', '')[:500]}..."
                for content in valid_contents
            ])
            
            # 5. Create CrewAI Agents
            researcher = Agent(
                role='Senior Research Analyst',
                goal='Systematically analyze research materials and extract critical insights',
                backstory='A meticulous researcher with expertise in synthesizing complex information from multiple sources',
                verbose=True,
                llm=self.mistral_llm
            )
            
            synthesizer = Agent(
                role='Research Synthesizer',
                goal='Transform raw research findings into a coherent, structured summary',
                backstory='An expert communicator who can distill multifaceted research into clear, actionable insights',
                verbose=True,
                llm=self.mistral_llm
            )
            
            # 6. Define Research Tasks
            research_task = Task(
                description=f"""Comprehensively analyze research materials on the topic: '{query}'
                Critical requirements:
                - Prioritise Latest Information
                - Extract precise insights and recent developments
                - Maintain strict adherence to source materials
                - Provide evidence-based analysis""",
                agent=researcher,
                expected_output="Detailed research analysis with key findings and insights"
            )
            
            synthesis_task = Task(
                description="""Based on the research analysis:
                - Create a cohesive, well-structured summary
                - Ensure clarity and logical flow of information
                - Maintain objectivity and reflect source material accurately
                - Highlight the most significant findings
                - Format for readability and quick comprehension""",
                agent=synthesizer,
                expected_output="Polished, concise research summary capturing essential insights"
            )
            
            # 7. Execute Research Workflow
            crew = Crew(
                agents=[researcher, synthesizer],
                tasks=[research_task, synthesis_task],
                verbose=True
            )
            
            result = crew.kickoff(inputs={'research_context': research_context})
            
            # 8. Prepare Comprehensive Results
            workflow_end_time = datetime.datetime.now()
            workflow_duration = (workflow_end_time - workflow_start_time).total_seconds()
            
            return {
                'status': 'success',
                'query': query,
                'workflow_duration': workflow_duration,
                'search_results': search_results,
                'extracted_contents': valid_contents,
                'research_context': research_context,
                'final_research_summary': result,
                'metadata': {
                    'total_sources_retrieved': len(search_results),
                    'valid_sources_extracted': len(valid_contents),
                    'timestamp': workflow_end_time.isoformat()
                }
            }
        
        except Exception as workflow_error:
            logger.critical(f"Research workflow encountered a critical error: {workflow_error}")
            return {
                'status': 'critical_error',
                'message': str(workflow_error),
                'query': query
            }

# # Async main function for demonstration
# async def main():
#     # Initialize advanced research system
#     research_system = AdvancedResearchSystem()
    
#     # Conduct research
#     query = "Microsoft Stock Price Current"
#     results = await research_system.advanced_research_workflow(query)
    
#     # Display results
#     print("\nFinal Research Summary:")
#     print(results['final_research_summary'])

# # Run the research workflow
# if __name__ == "__main__":
#     asyncio.run(main())

# The main function can remain largely the same
async def main():
    research_system = AdvancedResearchSystem()
    
    # Conduct research with more detailed output
    query = "Syria Bashar al-Assad"
    results = await research_system.advanced_research_workflow(query)
    
    # Display comprehensive results
    print("\nSearch Results Details:")
    print("\nResults Of Process So Far: ", results)
    # print(f"Number of Sources: {len(results['search_results'])}")
    
    print("\nAgent Reasoning Logs:")
    # for agent_type, steps in results['agent_logs'].items():
    #     print(f"\n{agent_type.replace('_', ' ').title()}:")
    #     for step in steps:
    #         print(f"- {step['timestamp']}: {step['step']}")
    
    print("\nFinal Research Summary:")
    print(results['final_research_summary'])

# Run the research workflow
if __name__ == "__main__":
    asyncio.run(main())