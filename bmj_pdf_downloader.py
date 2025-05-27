"""BMJ Best Practice PDF downloader for cardiology guidelines."""
import requests
from bs4 import BeautifulSoup
import os
import time
import re
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

class BMJPDFDownloader:
    """Downloader for BMJ Best Practice cardiology PDFs."""
    
    def __init__(self, download_dir: str = "cardiology_pdfs"):
        """Initialize the downloader."""
        self.download_dir = download_dir
        self.base_url = "https://bestpractice.bmj.com"
        self.cardiology_url = "https://bestpractice.bmj.com/specialties/2/Cardiology"
        self.session = requests.Session()
        
        # Set user agent to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create download directory
        os.makedirs(download_dir, exist_ok=True)
    
    def get_cardiology_topics(self) -> List[Dict[str, str]]:
        """Extract all cardiology topic links from the main cardiology page."""
        print("Fetching cardiology topics from BMJ Best Practice...")
        
        try:
            response = self.session.get(self.cardiology_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching cardiology page: {e}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        topics = []
        
        # Based on the provided HTML structure, look for topic links
        # The topics are organized in sections A-Z
        sections = soup.find_all(['div', 'section'], class_=['section', 'topic-list'])
        
        # Alternative approach: find all links that appear to be topic links
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Filter for topic links (they typically contain '/topics/')
            if '/topics/' in href and text and len(text) > 3:
                # Skip navigation and general links
                if any(skip in text.lower() for skip in ['home', 'about', 'contact', 'login', 'subscribe']):
                    continue
                
                full_url = urljoin(self.base_url, href)
                topics.append({
                    'name': text,
                    'url': full_url,
                    'safe_name': self._sanitize_filename(text)
                })
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_topics = []
        for topic in topics:
            if topic['url'] not in seen_urls:
                seen_urls.add(topic['url'])
                unique_topics.append(topic)
        
        print(f"Found {len(unique_topics)} cardiology topics")
        return unique_topics
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize topic name for use as filename."""
        # Remove special characters and replace spaces with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
        sanitized = re.sub(r'\s+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized[:100]  # Limit length
    
    def get_pdf_link(self, topic_url: str) -> Optional[str]:
        """Extract PDF download link from a topic page."""
        try:
            response = self.session.get(topic_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching topic page {topic_url}: {e}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for PDF links - they might be in different formats
        pdf_patterns = [
            'a[href*=".pdf"]',
            'a[href*="pdf"]',
            'a:contains("PDF")',
            'a:contains("View PDF")',
            'a:contains("Download PDF")',
            '.pdf-link',
            '.download-pdf'
        ]
        
        for pattern in pdf_patterns:
            try:
                pdf_links = soup.select(pattern)
                for link in pdf_links:
                    href = link.get('href')
                    if href:
                        return urljoin(self.base_url, href)
            except:
                continue
        
        # Alternative: look for any link containing 'pdf' in text or href
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            text = link.get_text(strip=True).lower()
            
            if 'pdf' in href.lower() or 'pdf' in text:
                return urljoin(self.base_url, href)
        
        return None
    
    def download_pdf(self, pdf_url: str, filename: str) -> bool:
        """Download a single PDF file."""
        file_path = os.path.join(self.download_dir, f"{filename}.pdf")
        
        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"File already exists: {filename}.pdf")
            return True
        
        try:
            response = self.session.get(pdf_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check if response is actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not pdf_url.endswith('.pdf'):
                print(f"Warning: Response doesn't appear to be a PDF for {filename}")
                # Still try to save it in case it's a PDF with wrong content-type
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {filename}.pdf")
            return True
            
        except requests.RequestException as e:
            print(f"Error downloading PDF {filename}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error downloading {filename}: {e}")
            return False
    
    def download_all_cardiology_pdfs(self, delay: float = 1.0) -> Dict[str, List[str]]:
        """Download all available cardiology PDFs."""
        topics = self.get_cardiology_topics()
        
        if not topics:
            print("No topics found. You may need to adjust the scraping logic.")
            return {'success': [], 'failed': []}
        
        successful_downloads = []
        failed_downloads = []
        
        print(f"\nAttempting to download PDFs for {len(topics)} topics...")
        
        for topic in tqdm(topics, desc="Downloading PDFs"):
            topic_name = topic['name']
            topic_url = topic['url']
            safe_name = topic['safe_name']
            
            # Get PDF link
            pdf_url = self.get_pdf_link(topic_url)
            
            if pdf_url:
                success = self.download_pdf(pdf_url, safe_name)
                if success:
                    successful_downloads.append(topic_name)
                else:
                    failed_downloads.append(topic_name)
            else:
                print(f"No PDF found for: {topic_name}")
                failed_downloads.append(topic_name)
            
            # Add delay to be respectful to the server
            time.sleep(delay)
        
        # Generate summary
        summary = {
            'success': successful_downloads,
            'failed': failed_downloads,
            'total_topics': len(topics),
            'successful_count': len(successful_downloads),
            'failed_count': len(failed_downloads)
        }
        
        print(f"\nDownload Summary:")
        print(f"  Total topics: {summary['total_topics']}")
        print(f"  Successful downloads: {summary['successful_count']}")
        print(f"  Failed downloads: {summary['failed_count']}")
        
        if failed_downloads:
            print(f"\nFailed downloads:")
            for name in failed_downloads[:10]:  # Show first 10
                print(f"  - {name}")
            if len(failed_downloads) > 10:
                print(f"  ... and {len(failed_downloads) - 10} more")
        
        return summary
    
    def create_topic_list(self) -> None:
        """Create a text file with all available topics."""
        topics = self.get_cardiology_topics()
        
        list_file = os.path.join(self.download_dir, "cardiology_topics_list.txt")
        
        with open(list_file, 'w', encoding='utf-8') as f:
            f.write("BMJ Best Practice - Cardiology Topics\n")
            f.write("=" * 40 + "\n\n")
            
            for i, topic in enumerate(topics, 1):
                f.write(f"{i:3d}. {topic['name']}\n")
                f.write(f"     URL: {topic['url']}\n\n")
        
        print(f"Topic list saved to: {list_file}")

# Alternative approach using the known cardiology topics from the search results
class PredefinedCardiologyTopics:
    """Use predefined list of cardiology topics if web scraping fails."""
    
    CARDIOLOGY_TOPICS = [
        "Abdominal aortic aneurysm", "Acute heart failure", "Anticoagulation management principles",
        "Aortic coarctation", "Aortic dissection", "Aortic regurgitation", "Aortic stenosis",
        "Atrial flutter", "Atrial myxoma", "Atrioventricular block", "Bradycardia", "Brugada syndrome",
        "Cardiac arrest", "Cardiac tamponade", "Carotid artery stenosis", "Chronic coronary disease",
        "Chronic venous insufficiency", "Congenital heart disease", "Diabetic cardiovascular disease",
        "Digoxin overdose", "Essential hypertension", "Established atrial fibrillation",
        "Evaluation of cardiomyopathy", "Evaluation of chest pain", "Evaluation of clubbing",
        "Evaluation of cyanosis in the newborn", "Evaluation of dizziness", "Evaluation of hypertension",
        "Evaluation of hypotension", "Evaluation of mediastinal mass", "Evaluation of palpitations",
        "Evaluation of pericardial effusion", "Evaluation of peripheral edema", "Evaluation of shock",
        "Evaluation of syncope", "Evaluation of tachycardia", "Focal atrial tachycardia",
        "Gestational hypertension", "Heart failure with preserved ejection fraction",
        "Heart failure with reduced ejection fraction", "Heparin-induced thrombocytopenia",
        "Hypercholesterolemia", "Hypertensive emergencies", "Hypertriglyceridemia",
        "Hypertrophic cardiomyopathy", "Idiopathic pulmonary arterial hypertension",
        "Infective endocarditis", "Interatrial communications (atrial septal defects)",
        "Kawasaki disease", "Long QT syndrome", "Marfan syndrome", "Mitral regurgitation",
        "Mitral stenosis", "Myocarditis", "Neurally mediated reflex syncope",
        "New-onset atrial fibrillation", "Non-ST-elevation myocardial infarction",
        "Nonsustained ventricular tachycardias", "Orthostatic hypotension",
        "Overview of acute coronary syndrome", "Overview of dysrhythmias (cardiac)",
        "Patent ductus arteriosus", "Patent foramen ovale", "Pericarditis",
        "Peripheral arterial disease", "Postural orthostatic tachycardia syndrome",
        "Preoperative cardiac risk assessment", "Pulmonary regurgitation", "Pulmonary stenosis",
        "Raynaud phenomenon", "Renal artery stenosis", "Rheumatic fever", "Shock",
        "ST-elevation myocardial infarction", "Superior vena cava syndrome",
        "Sustained ventricular tachycardias", "Takayasu arteritis", "Tetralogy of Fallot",
        "Tricuspid regurgitation", "Tricuspid stenosis", "Unstable angina",
        "Ventricular septal defects", "Wolff-Parkinson-White syndrome"
    ]
    
    @classmethod
    def create_topic_urls(cls) -> List[Dict[str, str]]:
        """Create topic URLs from predefined list."""
        base_url = "https://bestpractice.bmj.com/topics/en-us/"
        topics = []
        
        for topic in cls.CARDIOLOGY_TOPICS:
            # Create URL-friendly version of topic name
            url_name = topic.lower().replace(' ', '-').replace('(', '').replace(')', '').replace(',', '')
            url_name = re.sub(r'[^a-z0-9-]', '', url_name)
            
            topics.append({
                'name': topic,
                'url': f"{base_url}{url_name}",
                'safe_name': re.sub(r'[<>:"/\\|?*]', '', topic).replace(' ', '_')[:100]
            })
        
        return topics

# Example usage
if __name__ == "__main__":
    downloader = BMJPDFDownloader()
    
    # Create topic list first
    print("Creating list of available topics...")
    downloader.create_topic_list()
    
    # Try to download PDFs
    print("\nStarting PDF downloads...")
    summary = downloader.download_all_cardiology_pdfs(delay=2.0)
    
    print(f"\nDownload completed!")
    print(f"Check the '{downloader.download_dir}' directory for downloaded PDFs.") 