import os
import re
import json
from bs4 import BeautifulSoup
from boilerpy3 import extractors
from tqdm import tqdm
import hashlib
import base64
from urllib.parse import urljoin

class AsyncAPIDocsPreprocessor:
    def __init__(self, input_dir='asyncapi-docs', output_dir='processed_asyncapi_docs'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.extractor = extractors.ArticleExtractor()
        os.makedirs(output_dir, exist_ok=True)
    
    def clean_html(self, soup, base_path):
        """Remove unnecessary elements while preserving code blocks and relevant images"""
        # Remove navigation and decorative elements
        for element in soup(['header', 'footer', 'nav', 'script', 'style', 'button', 'form']):
            element.decompose()
        
        # Remove divs with common classes used for navigation
        for div in soup.find_all('div', class_=re.compile(r'sidebar|toc|navigation|nav|menu')):
            div.decompose()
        
        # Process images - keep only informative ones (diagrams, screenshots)
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                # Check if image is likely to be content (not icon/logo)
                if not re.search(r'icon|logo|avatar', src, re.I):
                    # Convert to absolute path if relative
                    if not src.startswith(('http', 'data:')):
                        img['src'] = urljoin(base_path, src)
                    # Keep the image
                    continue
            img.decompose()
        
        return soup
    
    def extract_special_blocks(self, soup):
        """Extract and preserve code examples (YAML, JavaScript), images, and other special content"""
        special_blocks = {
            'code_blocks': [],
            'images': [],
            'yaml_blocks': []
        }
        
        # Process all code blocks
        for pre in soup.find_all('pre'):
            code = pre.get_text().strip()
            if len(code.split('\n')) > 1 or len(code) > 50:  # Only keep substantial code blocks
                # Check for YAML specifically
                if re.search(r'^\s*asyncapi:|^\s*[a-z]+:', code, re.I | re.M):
                    special_blocks['yaml_blocks'].append(code)
                    marker = f"YAML_BLOCK_{len(special_blocks['yaml_blocks'])-1}"
                else:
                    special_blocks['code_blocks'].append(code)
                    marker = f"CODE_BLOCK_{len(special_blocks['code_blocks'])-1}"
                pre.replace_with(f"[{marker}]")
        
        # Process images
        for img in soup.find_all('img'):
            src = img.get('src')
            if src:
                special_blocks['images'].append({
                    'src': src,
                    'alt': img.get('alt', '')
                })
                marker = f"IMAGE_BLOCK_{len(special_blocks['images'])-1}"
                img.replace_with(f"[{marker}]")
        
        return special_blocks
    
    def process_content(self, text, special_blocks):
        """Process content and reintegrate special blocks"""
        # Normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Reinsert code blocks with markers
        for i, code in enumerate(special_blocks['code_blocks']):
            text = text.replace(f"[CODE_BLOCK_{i}]", f"\n```javascript\n{code}\n```\n")
        
        # Reinsert YAML blocks with markers
        for i, yaml in enumerate(special_blocks['yaml_blocks']):
            text = text.replace(f"[YAML_BLOCK_{i}]", f"\n```yaml\n{yaml}\n```\n")
        
        # Reinsert images with markers
        for i, img in enumerate(special_blocks['images']):
            text = text.replace(
                f"[IMAGE_BLOCK_{i}]", 
                f"\n![{img['alt']}]({img['src']})\n"
            )
        
        return text
    
    def process_file(self, file_path):
        """Process a single documentation file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        # Get base path for resolving relative image URLs
        base_path = os.path.dirname(file_path) + '/'
        
        # Extract and clean
        soup = self.clean_html(soup, base_path)
        special_blocks = self.extract_special_blocks(soup)
        title = soup.title.string if soup.title else os.path.basename(file_path)
        content = soup.get_text(separator='\n', strip=True)
        processed_content = self.process_content(content, special_blocks)
        
        # Create document ID
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:8]
        
        return {
            'id': doc_id,
            'title': title,
            'path': file_path,
            'content': processed_content,
            'code_blocks': special_blocks['code_blocks'],
            'yaml_blocks': special_blocks['yaml_blocks'],
            'images': special_blocks['images'],
            'source': 'asyncapi-docs'
        }
    
    def run(self):
        """Process all HTML files and create a knowledge base"""
        html_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.html'):
                    html_files.append(os.path.join(root, file))
        
        knowledge_base = []
        for file_path in tqdm(html_files, desc="Processing AsyncAPI docs"):
            try:
                doc = self.process_file(file_path)
                knowledge_base.append(doc)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        # Save knowledge base
        output_path = os.path.join(self.output_dir, 'asyncapi_knowledge_base.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, indent=2)
        
        print(f"Processed {len(knowledge_base)} files. Knowledge base saved to {output_path}")
        return knowledge_base

if __name__ == "__main__":
    preprocessor = AsyncAPIDocsPreprocessor()
    knowledge_base = preprocessor.run()