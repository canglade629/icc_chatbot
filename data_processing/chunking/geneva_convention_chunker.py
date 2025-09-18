"""
Geneva Convention PDF Chunking Pipeline

This module implements a specialized PDF chunking pipeline for Geneva Convention
documents, supplementary protocols, statutes, and other law articles. It focuses
on preserving legal structure and article numbering for precise legal citation.

Key Features:
- Hierarchical splitting by Convention → Part → Section → Article → Paragraph
- Article-centric chunking (articles as atomic units)
- Preserved legal numbering and citations
- Specialized metadata for legal documents
- Optimized for legal document retrieval
"""

import fitz  # PyMuPDF
import re
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from section_classifier import LegalSectionClassifier, SectionType, SectionClassification
import pytesseract
from PIL import Image
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenevaChunkMetadata:
    """Enhanced metadata structure for Geneva Convention chunks"""
    doc_name: str
    part: str
    section: str
    article: str
    paragraph: Optional[str]
    sub_paragraph: Optional[str]
    pages: List[int]
    date: str
    source: str
    chunk_id: str
    article_type: str  # e.g., "common_article", "regular_article"
    section_type: str  # New: specific section type (e.g., "obligations")
    section_hierarchy: int  # New: hierarchical level (1-5)
    classification_confidence: float  # New: confidence score for section classification
    classification_evidence: List[str]  # New: evidence used for classification

@dataclass
class WordInfo:
    """Word information with positional metadata"""
    text: str
    bbox: Tuple[float, float, float, float]
    font_size: float
    font_name: str
    page_num: int

class GenevaConventionChunker:
    """
    Specialized chunker for Geneva Convention documents
    """
    
    def __init__(self, target_chunk_size: int = 600, overlap_ratio: float = 0.12):
        self.target_chunk_size = target_chunk_size
        self.overlap_ratio = overlap_ratio
        self.min_chunk_size = 200
        self.max_chunk_size = 800
        
        # Initialize section classifier
        self.section_classifier = LegalSectionClassifier()
        
        # Legal structural patterns for Geneva Conventions
        self.article_patterns = [
            r'^Article\s+(\d+[a-z]?)\s*[:\.]?\s*(.*)',
            r'^Art\.\s+(\d+[a-z]?)\s*[:\.]?\s*(.*)',
            r'^ARTICLE\s+(\d+[a-z]?)\s*[:\.]?\s*(.*)',
        ]
        
        self.part_patterns = [
            r'^Part\s+([IVXLC]+)\s*[:\.]?\s*(.*)',
            r'^PART\s+([IVXLC]+)\s*[:\.]?\s*(.*)',
            r'^Chapter\s+([IVXLC]+)\s*[:\.]?\s*(.*)',
        ]
        
        self.section_patterns = [
            r'^Section\s+([IVXLC]+)\s*[:\.]?\s*(.*)',
            r'^SECTION\s+([IVXLC]+)\s*[:\.]?\s*(.*)',
        ]
        
        self.paragraph_patterns = [
            r'^\((\d+)\)\s+(.*)',
            r'^\(([a-z])\)\s+(.*)',
            r'^\(([ivxlc]+)\)\s+(.*)',
        ]
        
        self.sub_paragraph_patterns = [
            r'^\((\d+[a-z])\)\s+(.*)',
            r'^\(([a-z]\d+)\)\s+(.*)',
        ]
        
        # Common article patterns (Article 3 common to all GCs)
        self.common_article_patterns = [
            r'Article\s+3\s+common\s+to\s+the\s+Geneva\s+Conventions',
            r'Common\s+Article\s+3',
            r'Article\s+3\s+\(common\)',
        ]
        
        # Document type detection
        self.doc_type_patterns = {
            'treaty': [r'Geneva\s+Convention', r'Convention\s+of\s+Geneva'],
            'protocol': [r'Protocol\s+Additional', r'Additional\s+Protocol'],
            'statute': [r'Statute', r'Rome\s+Statute'],
            'commentary': [r'Commentary', r'Comment'],
        }

    def extract_text_and_words(self, pdf_path: str) -> Tuple[str, List[WordInfo]]:
        """
        Extract text and word-level metadata from PDF using PyMuPDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (full_text, word_info_list)
        """
        doc = fitz.open(pdf_path)
        full_text = ""
        word_info_list = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text with layout preservation
            text = page.get_text()
            full_text += text + "\n"
            
            # Extract word-level information
            words = page.get_text("words")
            for word in words:
                word_info = WordInfo(
                    text=word[4],  # The actual text
                    bbox=word[:4],  # Bounding box coordinates
                    font_size=word[5] if len(word) > 5 else 12.0,
                    font_name=word[6] if len(word) > 6 else "unknown",
                    page_num=page_num + 1
                )
                word_info_list.append(word_info)
        
        doc.close()
        return full_text, word_info_list

    def extract_text_with_ocr(self, pdf_path: str) -> Tuple[str, List[WordInfo]]:
        """
        Extract text using OCR when regular text extraction fails
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (full_text, word_info_list)
        """
        logger.info(f"Using OCR fallback for: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = ""
        word_info_list = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            try:
                # Use Tesseract with custom config for better legal text recognition
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(img, config=custom_config)
                full_text += text + "\n"
                
                # Create basic word info for OCR text
                words = text.split()
                y_pos = 0
                for i, word in enumerate(words):
                    word_info = WordInfo(
                        text=word,
                        bbox=(0, y_pos, 100, y_pos + 20),  # Approximate positioning
                        font_size=12.0,
                        font_name="ocr",
                        page_num=page_num + 1
                    )
                    word_info_list.append(word_info)
                    y_pos += 25  # Approximate line spacing
                    
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                continue
        
        doc.close()
        return full_text, word_info_list

    def extract_text_with_fallback(self, pdf_path: str) -> Tuple[str, List[WordInfo]]:
        """
        Extract text with OCR fallback if regular extraction yields no meaningful content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (full_text, word_info_list)
        """
        # Try regular text extraction first
        full_text, word_info_list = self.extract_text_and_words(pdf_path)
        
        # Check if we got meaningful content
        meaningful_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # If text is too short or contains mostly non-alphabetic characters, use OCR
        if len(meaningful_text) < 100 or len(re.findall(r'[a-zA-Z]', meaningful_text)) < 50:
            logger.info(f"Regular text extraction yielded insufficient content, trying OCR for: {pdf_path}")
            return self.extract_text_with_ocr(pdf_path)
        
        return full_text, word_info_list

    def detect_headers_and_footers(self, words: List[WordInfo], threshold: float = 0.7) -> Tuple[List[str], List[str]]:
        """
        Detect repeated headers and footers using frequency analysis
        
        Args:
            words: List of WordInfo objects
            threshold: Frequency threshold for considering text as header/footer
            
        Returns:
            Tuple of (headers, footers)
        """
        # Group words by page
        page_words = defaultdict(list)
        for word in words:
            page_words[word.page_num].append(word)
        
        # Extract text from top and bottom of each page
        headers = []
        footers = []
        
        for page_num, page_word_list in page_words.items():
            if not page_word_list:
                continue
                
            # Sort by y-coordinate (top to bottom)
            sorted_words = sorted(page_word_list, key=lambda w: w.bbox[1])
            
            # Get top 15% and bottom 15% of words
            top_count = max(1, len(sorted_words) // 7)
            bottom_count = max(1, len(sorted_words) // 7)
            
            top_words = sorted_words[:top_count]
            bottom_words = sorted_words[-bottom_count:]
            
            # Extract text
            header_text = " ".join([w.text for w in top_words if w.text.strip()])
            footer_text = " ".join([w.text for w in bottom_words if w.text.strip()])
            
            if header_text.strip():
                headers.append(header_text.strip())
            if footer_text.strip():
                footers.append(footer_text.strip())
        
        # Find frequently occurring headers and footers
        header_counter = Counter(headers)
        footer_counter = Counter(footers)
        
        frequent_headers = [text for text, count in header_counter.items() 
                          if count >= len(page_words) * threshold]
        frequent_footers = [text for text, count in footer_counter.items() 
                          if count >= len(page_words) * threshold]
        
        return frequent_headers, frequent_footers

    def detect_table_of_contents(self, text: str) -> Tuple[bool, int, int]:
        """
        Detect and locate table of contents in the document
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_toc, start_line, end_line)
        """
        lines = text.split('\n')
        toc_patterns = [
            r'table\s+of\s+contents',
            r'contents',
            r'index',
            r'list\s+of\s+figures',
            r'list\s+of\s+tables'
        ]
        
        start_line = -1
        end_line = -1
        
        # Look for TOC in first 15% of document
        search_lines = int(len(lines) * 0.15)
        
        for i, line in enumerate(lines[:search_lines]):
            line_lower = line.lower().strip()
            
            # Check for TOC heading
            for pattern in toc_patterns:
                if re.search(pattern, line_lower):
                    start_line = i
                    break
            
            if start_line != -1:
                # Look for end of TOC (usually a page number or section heading)
                for j in range(i + 1, min(i + 50, len(lines))):
                    end_line = j
                    next_line = lines[j].strip()
                    
                    # End conditions
                    if (re.match(r'^[A-Z][A-Z\s]+$', next_line) or  # All caps heading
                        re.match(r'^\d+\.?\s+[A-Z]', next_line) or    # Numbered section
                        re.match(r'^Chapter\s+\d+', next_line, re.IGNORECASE) or
                        re.match(r'^Section\s+\d+', next_line, re.IGNORECASE) or
                        re.match(r'^Article\s+\d+', next_line, re.IGNORECASE) or
                        next_line == '' and j > i + 5):  # Empty line after TOC
                        break
                
                break
        
        return start_line != -1, start_line, end_line

    def remove_headers_footers(self, text: str, headers: List[str], footers: List[str]) -> str:
        """
        Remove detected headers and footers from text
        
        Args:
            text: Input text
            headers: List of header patterns to remove
            footers: List of footer patterns to remove
            
        Returns:
            Cleaned text
        """
        cleaned_text = text
        
        # Detect and remove table of contents
        is_toc, start_line, end_line = self.detect_table_of_contents(text)
        if is_toc and start_line != -1 and end_line != -1:
            lines = cleaned_text.split('\n')
            # Remove TOC lines
            lines = lines[:start_line] + lines[end_line:]
            cleaned_text = '\n'.join(lines)
            logger.info(f"Removed table of contents (lines {start_line}-{end_line})")
        
        # Remove headers
        for header in headers:
            escaped_header = re.escape(header)
            cleaned_text = re.sub(escaped_header, '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove footers
        for footer in footers:
            escaped_footer = re.escape(footer)
            cleaned_text = re.sub(escaped_footer, '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        return cleaned_text.strip()

    def detect_document_type(self, text: str, filename: str) -> str:
        """
        Detect the type of legal document
        
        Args:
            text: Document text
            filename: PDF filename
            
        Returns:
            Document type (treaty, protocol, statute, commentary)
        """
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        for doc_type, patterns in self.doc_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower) or re.search(pattern, filename_lower):
                    return doc_type
        
        return "unknown"

    def extract_document_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Extract document-level metadata
        
        Args:
            text: Document text
            filename: PDF filename
            
        Returns:
            Dictionary of document metadata
        """
        metadata = {
            'doc_name': filename.replace('.pdf', ''),
            'date': '',
            'source': self.detect_document_type(text, filename),
            'filename': filename
        }
        
        # Extract date patterns
        date_patterns = [
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}\.\d{1,2}\.\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['date'] = match.group(1)
                break
        
        # Extract document name from text
        title_patterns = [
            r'Geneva\s+Convention\s+[IVXLC]+',
            r'Protocol\s+Additional\s+[IVXLC]+',
            r'Rome\s+Statute',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['doc_name'] = match.group(0)
                break
        
        return metadata

    def classify_section_enhanced(self, text: str, context: Optional[Dict[str, Any]] = None) -> SectionClassification:
        """
        Enhanced section classification using the new classifier
        
        Args:
            text: Text to classify
            context: Optional context information
            
        Returns:
            SectionClassification object
        """
        return self.section_classifier.classify_section(text, 'geneva', context)

    def split_by_articles(self, text: str) -> List[Dict[str, Any]]:
        """
        Enhanced split into article blocks with intelligent section classification
        
        Args:
            text: Input text
            
        Returns:
            List of article blocks with enhanced metadata
        """
        articles = []
        lines = text.split('\n')
        
        current_article = None
        current_part = ""
        current_section = ""
        previous_section_type = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for part markers
            for pattern in self.part_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    current_part = f"Part {match.group(1)}"
                    if match.group(2).strip():
                        current_part += f": {match.group(2).strip()}"
                    continue
            
            # Check for section markers
            for pattern in self.section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    current_section = f"Section {match.group(1)}"
                    if match.group(2).strip():
                        current_section += f": {match.group(2).strip()}"
                    continue
            
            # Check for article markers
            for pattern in self.article_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous article if exists
                    if current_article and current_article['text'].strip():
                        # Classify the article before saving
                        classification = self.classify_section_enhanced(
                            current_article['text'], 
                            {'previous_section_type': previous_section_type}
                        )
                        current_article['section_classification'] = classification
                        articles.append(current_article)
                    
                    # Start new article
                    article_num = match.group(1)
                    article_title = match.group(2).strip() if len(match.groups()) > 1 else ""
                    
                    # Determine article type
                    article_type = "common_article" if self._is_common_article(article_num, article_title) else "regular_article"
                    
                    current_article = {
                        'text': line,
                        'article_num': article_num,
                        'article_title': article_title,
                        'part': current_part,
                        'section': current_section,
                        'article_type': article_type,
                        'paragraphs': [],
                        'page_start': 1,  # Will be updated with actual page info
                        'page_end': 1,
                        'section_classification': None
                    }
                    break
            
            # If we're in an article, add content
            if current_article:
                # Check for paragraph markers within article
                paragraph_found = False
                for pattern in self.paragraph_patterns:
                    match = re.match(pattern, line)
                    if match:
                        paragraph_num = match.group(1)
                        paragraph_text = match.group(2).strip()
                        current_article['paragraphs'].append({
                            'num': paragraph_num,
                            'text': paragraph_text
                        })
                        paragraph_found = True
                        break
                
                if not paragraph_found:
                    # Add line to article text
                    if current_article['text']:
                        current_article['text'] += '\n' + line
                    else:
                        current_article['text'] = line
        
        # Don't forget the last article
        if current_article and current_article['text'].strip():
            # Classify the final article
            classification = self.classify_section_enhanced(
                current_article['text'], 
                {'previous_section_type': previous_section_type}
            )
            current_article['section_classification'] = classification
            articles.append(current_article)
        
        # Update previous section type for context
        for article in articles:
            if article.get('section_classification'):
                previous_section_type = article['section_classification'].section_type.value
        
        return articles

    def _is_common_article(self, article_num: str, article_title: str) -> bool:
        """Check if this is a common article (like Article 3)"""
        if article_num == "3":
            return True
        
        title_lower = article_title.lower()
        for pattern in self.common_article_patterns:
            if re.search(pattern, title_lower):
                return True
        
        return False

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate using word count)
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        try:
            words = word_tokenize(text)
            return len(words)
        except Exception:
            # Fallback to simple word splitting
            return len(text.split())

    def split_article_content(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split article content into chunks if needed
        
        Args:
            article: Article dictionary
            
        Returns:
            List of chunks for this article
        """
        chunks = []
        
        # If article is short enough, keep as single chunk
        if self.count_tokens(article['text']) <= self.target_chunk_size:
            chunks.append({
                'text': article['text'],
                'article_num': article['article_num'],
                'article_title': article['article_title'],
                'part': article['part'],
                'section': article['section'],
                'article_type': article['article_type'],
                'paragraph': None,
                'sub_paragraph': None,
                'tokens': self.count_tokens(article['text']),
                'split_method': 'article'
            })
        else:
            # Split by paragraphs if available
            if article['paragraphs']:
                for para in article['paragraphs']:
                    para_text = f"Article {article['article_num']}: {article['article_title']}\n\n({para['num']}) {para['text']}"
                    
                    if self.count_tokens(para_text) <= self.target_chunk_size:
                        chunks.append({
                            'text': para_text,
                            'article_num': article['article_num'],
                            'article_title': article['article_title'],
                            'part': article['part'],
                            'section': article['section'],
                            'article_type': article['article_type'],
                            'paragraph': para['num'],
                            'sub_paragraph': None,
                            'tokens': self.count_tokens(para_text),
                            'split_method': 'paragraph'
                        })
                    else:
                        # Split by sentences
                        sentence_chunks = self._split_by_sentences(para_text, article)
                        chunks.extend(sentence_chunks)
            else:
                # Split by sentences
                sentence_chunks = self._split_by_sentences(article['text'], article)
                chunks.extend(sentence_chunks)
        
        return chunks

    def _split_by_sentences(self, text: str, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text by sentences when articles/paragraphs are too long"""
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = text.split('. ')
            sentences = [s.strip() + '.' if not s.endswith('.') else s.strip() for s in sentences]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + (' ' if current_chunk else '') + sentence
            if self.count_tokens(test_chunk) <= self.target_chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'article_num': article['article_num'],
                        'article_title': article['article_title'],
                        'part': article['part'],
                        'section': article['section'],
                        'article_type': article['article_type'],
                        'paragraph': None,
                        'sub_paragraph': None,
                        'tokens': self.count_tokens(current_chunk),
                        'split_method': 'sentence'
                    })
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'article_num': article['article_num'],
                'article_title': article['article_title'],
                'part': article['part'],
                'section': article['section'],
                'article_type': article['article_type'],
                'paragraph': None,
                'sub_paragraph': None,
                'tokens': self.count_tokens(current_chunk),
                'split_method': 'sentence'
            })
        
        return chunks

    def add_overlap(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add overlap between adjacent chunks"""
        if len(chunks) <= 1:
            return chunks
        
        chunks_with_overlap = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                chunks_with_overlap.append(chunk)
                continue
            
            # Calculate overlap tokens
            overlap_tokens = int(chunk['tokens'] * self.overlap_ratio)
            
            # Get previous chunk text
            prev_text = chunks[i-1]['text']
            prev_words = prev_text.split()
            
            # Get current chunk text
            current_words = chunk['text'].split()
            
            # Add overlap from previous chunk
            if overlap_tokens > 0 and len(prev_words) >= overlap_tokens:
                overlap_text = ' '.join(prev_words[-overlap_tokens:])
                overlapped_text = overlap_text + ' ' + chunk['text']
                
                overlapped_chunk = chunk.copy()
                overlapped_chunk['text'] = overlapped_text
                overlapped_chunk['tokens'] = self.count_tokens(overlapped_text)
                overlapped_chunk['overlap_tokens'] = overlap_tokens
                
                chunks_with_overlap.append(overlapped_chunk)
            else:
                chunks_with_overlap.append(chunk)
        
        return chunks_with_overlap

    def build_chunks(self, doc_text: str, articles: List[Dict[str, Any]], 
                    metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build final chunks with complete metadata
        
        Args:
            doc_text: Full document text
            articles: List of article blocks
            metadata: Document metadata
            
        Returns:
            List of final chunks with metadata
        """
        all_chunks = []
        chunk_id = 0
        
        for article in articles:
            # Split article content
            article_chunks = self.split_article_content(article)
            
            # Apply small chunk merging (respecting article boundaries)
            article_chunks = self.merge_small_chunks(article_chunks)
            
            # Add overlap
            article_chunks = self.add_overlap(article_chunks)
            
            # Create chunk metadata
            for chunk in article_chunks:
                # Get section classification information
                section_classification = article.get('section_classification')
                if section_classification:
                    section_type = section_classification.section_type.value
                    section_hierarchy = section_classification.confidence
                    classification_confidence = section_classification.confidence
                    classification_evidence = section_classification.evidence
                else:
                    section_type = 'unknown'
                    section_hierarchy = 3
                    classification_confidence = 0.0
                    classification_evidence = []
                
                chunk_metadata = GenevaChunkMetadata(
                    doc_name=metadata['doc_name'],
                    part=chunk['part'],
                    section=chunk['section'],
                    article=chunk['article_num'],
                    paragraph=chunk['paragraph'],
                    sub_paragraph=chunk['sub_paragraph'],
                    pages=[article.get('page_start', 1), article.get('page_end', 1)],
                    date=metadata['date'],
                    source=metadata['source'],
                    chunk_id=f"{metadata['doc_name']}_chunk_{chunk_id}",
                    article_type=chunk['article_type'],
                    section_type=section_type,
                    section_hierarchy=section_hierarchy,
                    classification_confidence=classification_confidence,
                    classification_evidence=classification_evidence
                )
                
                all_chunks.append({
                    'text': chunk['text'],
                    'metadata': chunk_metadata,
                    'tokens': chunk['tokens'],
                    'split_method': chunk['split_method']
                })
                
                chunk_id += 1
        
        return all_chunks

    def merge_small_chunks(self, chunks: List[Dict[str, Any]], 
                          min_size: int = None, target_size: int = None, 
                          max_size: int = None) -> List[Dict[str, Any]]:
        """
        Merge adjacent small chunks to reduce fragmentation
        Don't merge across article boundaries for legal documents
        
        Args:
            chunks: List of chunks
            min_size: Minimum chunk size before merging
            target_size: Target size for merged chunks
            max_size: Maximum chunk size
            
        Returns:
            List of merged chunks
        """
        if min_size is None:
            min_size = self.min_chunk_size
        if target_size is None:
            target_size = self.target_chunk_size
        if max_size is None:
            max_size = self.max_chunk_size
        
        merged = []
        buffer = []
        buffer_tokens = 0
        current_article = None
        
        for chunk in chunks:
            tokens = chunk['tokens']
            chunk_article = chunk.get('article_num', '')
            
            # Check if we're crossing article boundaries
            if current_article and chunk_article and chunk_article != current_article:
                # Flush buffer before crossing article boundary
                if buffer:
                    merged_chunk = self._join_chunks(buffer)
                    merged.append(merged_chunk)
                    buffer, buffer_tokens = [], 0
                current_article = chunk_article
            elif not current_article:
                current_article = chunk_article
            
            if tokens < min_size:
                # Add to buffer
                buffer.append(chunk)
                buffer_tokens += tokens
                
                # If buffer is big enough OR we have too many small chunks, flush it
                if buffer_tokens >= target_size or len(buffer) >= 3:
                    merged_chunk = self._join_chunks(buffer)
                    merged.append(merged_chunk)
                    buffer, buffer_tokens = [], 0
            else:
                # Flush buffer first if it has content
                if buffer:
                    merged_chunk = self._join_chunks(buffer)
                    merged.append(merged_chunk)
                    buffer, buffer_tokens = [], 0
                
                # Handle the current chunk
                if tokens > max_size:
                    # Split large chunk further
                    split_chunks = self._split_large_chunk(chunk, max_size)
                    merged.extend(split_chunks)
                else:
                    merged.append(chunk)
        
        # Flush any remaining buffer
        if buffer:
            merged_chunk = self._join_chunks(buffer)
            merged.append(merged_chunk)
        
        return merged

    def _join_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Join multiple chunks into one"""
        if len(chunks) == 1:
            return chunks[0]
        
        # Combine text
        combined_text = '\n\n'.join([chunk['text'] for chunk in chunks])
        combined_tokens = sum([chunk['tokens'] for chunk in chunks])
        
        # Use metadata from first chunk as base
        base_chunk = chunks[0].copy()
        base_chunk['text'] = combined_text
        base_chunk['tokens'] = combined_tokens
        base_chunk['split_method'] = 'merged'
        
        return base_chunk

    def _split_large_chunk(self, chunk: Dict[str, Any], max_size: int) -> List[Dict[str, Any]]:
        """Split a chunk that's too large"""
        text = chunk['text']
        sentences = sent_tokenize(text) if len(text.split()) > 50 else [text]
        
        split_chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            test_chunk = current_chunk + (' ' if current_chunk else '') + sentence
            test_tokens = self.count_tokens(test_chunk)
            
            if test_tokens <= max_size:
                current_chunk = test_chunk
                current_tokens = test_tokens
            else:
                if current_chunk:
                    new_chunk = chunk.copy()
                    new_chunk['text'] = current_chunk
                    new_chunk['tokens'] = current_tokens
                    new_chunk['split_method'] = 'split_large'
                    split_chunks.append(new_chunk)
                
                current_chunk = sentence
                current_tokens = self.count_tokens(sentence)
        
        if current_chunk:
            new_chunk = chunk.copy()
            new_chunk['text'] = current_chunk
            new_chunk['tokens'] = current_tokens
            new_chunk['split_method'] = 'split_large'
            split_chunks.append(new_chunk)
        
        return split_chunks

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a single PDF file through the complete pipeline
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of processed chunks
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text and words with OCR fallback
        text, words = self.extract_text_with_fallback(pdf_path)
        
        # Detect headers and footers
        headers, footers = self.detect_headers_and_footers(words)
        logger.info(f"Detected {len(headers)} headers and {len(footers)} footers")
        
        # Remove headers and footers
        cleaned_text = self.remove_headers_footers(text, headers, footers)
        
        # Extract document metadata
        filename = Path(pdf_path).name
        metadata = self.extract_document_metadata(cleaned_text, filename)
        
        # Split by articles
        articles = self.split_by_articles(cleaned_text)
        logger.info(f"Found {len(articles)} articles")
        
        # Build final chunks
        chunks = self.build_chunks(cleaned_text, articles, metadata)
        logger.info(f"Created {len(chunks)} final chunks")
        
        return chunks

    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all processed chunks
        """
        directory = Path(directory_path)
        pdf_files = list(directory.glob("*.pdf"))
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            try:
                chunks = self.process_pdf(str(pdf_file))
                all_chunks.extend(chunks)
                logger.info(f"Processed {pdf_file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
                continue
        
        return all_chunks

    def process_multiple_directories(self, directory_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process all PDFs in multiple directories
        
        Args:
            directory_paths: List of paths to directories containing PDFs
            
        Returns:
            List of all processed chunks
        """
        all_chunks = []
        
        for directory_path in directory_paths:
            logger.info(f"Processing directory: {directory_path}")
            chunks = self.process_directory(directory_path)
            all_chunks.extend(chunks)
            logger.info(f"Total chunks so far: {len(all_chunks)}")
        
        return all_chunks

    def save_to_parquet(self, chunks: List[Dict[str, Any]], output_path: str):
        """
        Save chunks to parquet format
        
        Args:
            chunks: List of chunks
            output_path: Output file path
        """
        # Convert chunks to DataFrame format
        data = []
        for chunk in chunks:
            metadata = chunk['metadata']
            data.append({
                'chunk_id': metadata.chunk_id,
                'text': chunk['text'],
                'doc_name': metadata.doc_name,
                'part': metadata.part,
                'section': metadata.section,
                'article': metadata.article,
                'paragraph': metadata.paragraph,
                'sub_paragraph': metadata.sub_paragraph,
                'pages': metadata.pages,
                'date': metadata.date,
                'source': metadata.source,
                'article_type': metadata.article_type,
                'section_type': metadata.section_type,
                'section_hierarchy': metadata.section_hierarchy,
                'classification_confidence': metadata.classification_confidence,
                'classification_evidence': metadata.classification_evidence,
                'tokens': chunk['tokens'],
                'split_method': chunk['split_method']
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    """Main function to run the Geneva Convention chunking pipeline"""
    print("=" * 80)
    print("Geneva Convention PDF Chunking Pipeline")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize chunker
    chunker = GenevaConventionChunker(target_chunk_size=600, overlap_ratio=0.12)
    
    # Process all PDFs in the documentation directories
    input_directories = [
        "/Users/christophe.anglade/Documents/icc_chatbot/data/AI IHL /documentation",
        "/Users/christophe.anglade/Documents/icc_chatbot/data/AI IHL /documentation/commentary"
    ]
    output_path = "/Users/christophe.anglade/Documents/icc_chatbot/output/geneva_convention_chunks.parquet"
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get list of all PDF files
    all_pdf_files = []
    for directory in input_directories:
        pdf_files = list(Path(directory).glob("*.pdf"))
        all_pdf_files.extend(pdf_files)
    
    total_files = len(all_pdf_files)
    
    print(f"Found {total_files} PDF files to process")
    print(f"  - Main documentation: {len(list(Path(input_directories[0]).glob('*.pdf')))} files")
    print(f"  - Commentary: {len(list(Path(input_directories[1]).glob('*.pdf')))} files")
    print(f"Output will be saved to: {output_path}")
    print()
    
    all_chunks = []
    processed_files = 0
    start_time = time.time()
    
    for i, pdf_file in enumerate(all_pdf_files, 1):
        print(f"[{i}/{total_files}] Processing: {pdf_file.name}")
        print("-" * 60)
        
        try:
            file_start_time = time.time()
            chunks = chunker.process_pdf(str(pdf_file))
            file_end_time = time.time()
            
            all_chunks.extend(chunks)
            processed_files += 1
            
            print(f"✓ Successfully processed {pdf_file.name}")
            print(f"  -> Created {len(chunks)} chunks")
            print(f"  -> Processing time: {file_end_time - file_start_time:.2f} seconds")
            print(f"  -> Total chunks so far: {len(all_chunks)}")
            
            # Show sample chunk
            if chunks:
                sample_text = chunks[0]['text'][:150] + "..." if len(chunks[0]['text']) > 150 else chunks[0]['text']
                print(f"  -> Sample chunk: {sample_text}")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file.name}: {str(e)}")
            continue
        
        print()
    
    # Final save
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Processed {processed_files}/{total_files} files successfully")
    print(f"Total chunks created: {len(all_chunks)}")
    
    if all_chunks:
        print(f"💾 Saving final results to {output_path}...")
        chunker.save_to_parquet(all_chunks, output_path)
        print(f"✓ Final results saved successfully!")
        
        # Show statistics
        print()
        print("STATISTICS:")
        print(f"  - Average chunks per file: {len(all_chunks) / processed_files:.1f}")
        print(f"  - Total processing time: {time.time() - start_time:.2f} seconds")
        print(f"  - Average time per file: {(time.time() - start_time) / processed_files:.2f} seconds")
        
        # Show sample chunks
        print()
        print("SAMPLE CHUNKS:")
        for i, chunk in enumerate(all_chunks[:3]):
            print(f"Chunk {i+1}:")
            print(f"  Text: {chunk['text'][:200]}...")
            print(f"  Metadata: {chunk['metadata']}")
            print()
    else:
        print("❌ No chunks were created!")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    import time
    from datetime import datetime
    main()
