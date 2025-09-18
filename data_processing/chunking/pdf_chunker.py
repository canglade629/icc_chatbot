"""
ICC Judgment PDF Chunking Pipeline

This module implements a comprehensive PDF chunking pipeline specifically designed
for ICC (International Criminal Court) judgments. It handles both born-digital
and scanned PDFs, extracting structured legal content with proper metadata.

Key Features:
- PDF parsing with positional metadata using PyMuPDF
- Header/footer removal and footnote detection
- Structural splitting based on legal markers
- Recursive/semantic chunking with overlap
- Comprehensive metadata extraction
- Output to parquet format for vector DB integration
"""

import fitz  # PyMuPDF
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
class ChunkMetadata:
    """Enhanced metadata structure for each chunk"""
    doc_id: str
    case_number: str
    pages: List[int]
    section: str
    section_type: str  # New: specific section type (e.g., "factual_findings")
    section_hierarchy: int  # New: hierarchical level (1-5)
    footnotes: List[str]
    date: str
    judges: List[str]
    char_offset: Tuple[int, int]
    chunk_id: str
    parent_section: str
    font_info: Dict[str, Any]
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

class ICCJudgmentChunker:
    """
    Main class for chunking ICC judgment PDFs
    """
    
    def __init__(self, target_chunk_size: int = 700, overlap_ratio: float = 0.18):
        self.target_chunk_size = target_chunk_size
        self.overlap_ratio = overlap_ratio
        self.semantic_threshold = 0.88
        self.min_chunk_size = 200
        self.max_chunk_size = 1000
        
        # Initialize section classifier
        self.section_classifier = LegalSectionClassifier()
        
        # Legal structural patterns
        self.heading_patterns = [
            r'^JUDGMENT\s*$',
            r'^FINDINGS\s*$',
            r'^ORDER\s*$',
            r'^DECISION\s*$',
            r'^OPINION\s*$',
            r'^DISSENTING\s+OPINION\s*$',
            r'^SEPARATE\s+OPINION\s*$',
            r'^CONCURRING\s+OPINION\s*$',
        ]
        
        self.paragraph_patterns = [
            r'^¶\s*\d+',
            r'^Article\s+\d+',
            r'^Section\s+\d+(\.\d+)*',
            r'^\d+(\.\d+)*\s*$',
            r'^\(\d+\)',
            r'^\([a-z]\)',
        ]
        
        self.case_number_patterns = [
            r'ICC-\d+/\d+-\d+/\d+-\d+',
            r'ICC-\d+/\d+-\d+',
            r'Case\s+No\.?\s*ICC-\d+/\d+-\d+',
        ]
        
        self.date_patterns = [
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}',
        ]
        
        self.judge_patterns = [
            r'Judge\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'Presiding\s+Judge\s+[A-Z][a-z]+',
        ]

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
        total_pages = len(doc)
        
        logger.info(f"Processing {total_pages} pages with OCR...")
        
        for page_num in range(total_pages):
            if page_num % 10 == 0:  # Log progress every 10 pages
                logger.info(f"OCR progress: {page_num + 1}/{total_pages} pages")
            
            page = doc[page_num]
            
            # Convert page to image with lower resolution for speed
            mat = fitz.Matrix(1.5, 1.5)  # Reduced zoom for better performance
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            try:
                # Use Tesseract with basic config
                text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
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
        logger.info(f"OCR completed. Extracted {len(full_text)} characters from {total_pages} pages")
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
            
            # Get top 20% and bottom 20% of words
            top_count = max(1, len(sorted_words) // 5)
            bottom_count = max(1, len(sorted_words) // 5)
            
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

    def detect_footnotes(self, words: List[WordInfo]) -> List[Dict[str, Any]]:
        """
        Detect footnotes based on font size and position
        
        Args:
            words: List of WordInfo objects
            
        Returns:
            List of footnote dictionaries with metadata
        """
        footnotes = []
        
        # Group words by page
        page_words = defaultdict(list)
        for word in words:
            page_words[word.page_num].append(word)
        
        for page_num, page_word_list in page_words.items():
            if not page_word_list:
                continue
                
            # Sort by y-coordinate
            sorted_words = sorted(page_word_list, key=lambda w: w.bbox[1])
            
            # Calculate average font size
            font_sizes = [w.font_size for w in page_word_list if w.font_size > 0]
            avg_font_size = np.mean(font_sizes) if font_sizes else 12.0
            
            # Find words with significantly smaller font size (likely footnotes)
            footnote_threshold = avg_font_size * 0.8
            
            footnote_words = []
            for word in sorted_words:
                if word.font_size < footnote_threshold and word.text.strip():
                    footnote_words.append(word)
            
            # Group consecutive footnote words
            if footnote_words:
                current_footnote = []
                for i, word in enumerate(footnote_words):
                    if not current_footnote:
                        current_footnote.append(word)
                    else:
                        # Check if this word is close to the previous one
                        prev_word = current_footnote[-1]
                        if (word.bbox[1] - prev_word.bbox[3]) < 20:  # Within 20 pixels vertically
                            current_footnote.append(word)
                        else:
                            # Save current footnote and start new one
                            if current_footnote:
                                footnote_text = " ".join([w.text for w in current_footnote])
                                footnotes.append({
                                    'text': footnote_text,
                                    'page': page_num,
                                    'bbox': (
                                        min(w.bbox[0] for w in current_footnote),
                                        min(w.bbox[1] for w in current_footnote),
                                        max(w.bbox[2] for w in current_footnote),
                                        max(w.bbox[3] for w in current_footnote)
                                    ),
                                    'font_size': np.mean([w.font_size for w in current_footnote])
                                })
                            current_footnote = [word]
                
                # Don't forget the last footnote
                if current_footnote:
                    footnote_text = " ".join([w.text for w in current_footnote])
                    footnotes.append({
                        'text': footnote_text,
                        'page': page_num,
                        'bbox': (
                            min(w.bbox[0] for w in current_footnote),
                            min(w.bbox[1] for w in current_footnote),
                            max(w.bbox[2] for w in current_footnote),
                            max(w.bbox[3] for w in current_footnote)
                        ),
                        'font_size': np.mean([w.font_size for w in current_footnote])
                    })
        
        return footnotes

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
            # Escape special regex characters
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

    def classify_section_enhanced(self, text: str, context: Optional[Dict[str, Any]] = None) -> SectionClassification:
        """
        Enhanced section classification using the new classifier
        
        Args:
            text: Text to classify
            context: Optional context information
            
        Returns:
            SectionClassification object
        """
        return self.section_classifier.classify_section(text, 'icc', context)

    def structural_split(self, text: str) -> List[Dict[str, Any]]:
        """
        Enhanced structural split with intelligent section classification
        
        Args:
            text: Input text
            
        Returns:
            List of structural blocks with enhanced metadata
        """
        blocks = []
        lines = text.split('\n')
        
        current_block = {
            'text': '',
            'type': 'paragraph',
            'section_id': '',
            'page_start': 1,
            'page_end': 1,
            'section_classification': None
        }
        
        previous_section_type = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_block['text'].strip():
                    current_block['text'] += '\n'
                continue
            
            # Check for headings first
            is_heading = False
            for pattern in self.heading_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save current block if it has content
                    if current_block['text'].strip():
                        # Classify the current block before saving
                        classification = self.classify_section_enhanced(
                            current_block['text'], 
                            {'previous_section_type': previous_section_type}
                        )
                        current_block['section_classification'] = classification
                        blocks.append(current_block.copy())
                    
                    # Start new heading block
                    current_block = {
                        'text': line,
                        'type': 'heading',
                        'section_id': line.upper().replace(' ', '_'),
                        'page_start': 1,  # Will be updated with actual page info
                        'page_end': 1,
                        'section_classification': None
                    }
                    is_heading = True
                    break
            
            if is_heading:
                continue
            
            # Check for paragraph markers
            is_paragraph_marker = False
            for pattern in self.paragraph_patterns:
                if re.match(pattern, line):
                    # Save current block if it has content
                    if current_block['text'].strip():
                        # Classify the current block before saving
                        classification = self.classify_section_enhanced(
                            current_block['text'], 
                            {'previous_section_type': previous_section_type}
                        )
                        current_block['section_classification'] = classification
                        blocks.append(current_block.copy())
                    
                    # Start new paragraph block
                    current_block = {
                        'text': line,
                        'type': 'paragraph',
                        'section_id': line,
                        'page_start': 1,
                        'page_end': 1,
                        'section_classification': None
                    }
                    is_paragraph_marker = True
                    break
            
            if is_paragraph_marker:
                continue
            
            # Add line to current block
            if current_block['text']:
                current_block['text'] += '\n' + line
            else:
                current_block['text'] = line
        
        # Don't forget the last block
        if current_block['text'].strip():
            # Classify the final block
            classification = self.classify_section_enhanced(
                current_block['text'], 
                {'previous_section_type': previous_section_type}
            )
            current_block['section_classification'] = classification
            blocks.append(current_block)
        
        # Update previous section type for context
        for block in blocks:
            if block.get('section_classification'):
                previous_section_type = block['section_classification'].section_type.value
        
        return blocks

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

    def recursive_split(self, text: str, max_tokens: int = None, overlap: float = None) -> List[Dict[str, Any]]:
        """
        Recursively split text into chunks with overlap
        
        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk
            overlap: Overlap ratio between chunks
            
        Returns:
            List of chunks with metadata
        """
        if max_tokens is None:
            max_tokens = self.target_chunk_size
        if overlap is None:
            overlap = self.overlap_ratio
        
        chunks = []
        
        # First try splitting by paragraphs
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph would exceed max_tokens
            test_chunk = current_chunk + ('\n\n' if current_chunk else '') + para
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': self.count_tokens(current_chunk),
                        'split_method': 'paragraph'
                    })
                
                # If single paragraph is too long, split by sentences
                if self.count_tokens(para) > max_tokens:
                    sentence_chunks = self._split_by_sentences(para, max_tokens, overlap)
                    chunks.extend(sentence_chunks)
                    current_chunk = ""
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': self.count_tokens(current_chunk),
                'split_method': 'paragraph'
            })
        
        # Add overlap between chunks
        chunks_with_overlap = self._add_overlap(chunks, overlap)
        
        return chunks_with_overlap

    def _split_by_sentences(self, text: str, max_tokens: int, overlap: float) -> List[Dict[str, Any]]:
        """Split text by sentences when paragraphs are too long"""
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
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'tokens': self.count_tokens(current_chunk),
                        'split_method': 'sentence'
                    })
                current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'tokens': self.count_tokens(current_chunk),
                'split_method': 'sentence'
            })
        
        return chunks

    def _add_overlap(self, chunks: List[Dict[str, Any]], overlap_ratio: float) -> List[Dict[str, Any]]:
        """Add overlap between adjacent chunks"""
        if len(chunks) <= 1:
            return chunks
        
        chunks_with_overlap = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                chunks_with_overlap.append(chunk)
                continue
            
            # Calculate overlap tokens
            overlap_tokens = int(chunk['tokens'] * overlap_ratio)
            
            # Get previous chunk text
            prev_text = chunks[i-1]['text']
            prev_words = word_tokenize(prev_text)
            
            # Get current chunk text
            current_words = word_tokenize(chunk['text'])
            
            # Add overlap from previous chunk
            if overlap_tokens > 0 and len(prev_words) >= overlap_tokens:
                overlap_text = ' '.join(prev_words[-overlap_tokens:])
                overlapped_text = overlap_text + ' ' + chunk['text']
                
                chunks_with_overlap.append({
                    'text': overlapped_text,
                    'tokens': self.count_tokens(overlapped_text),
                    'split_method': chunk['split_method'],
                    'overlap_tokens': overlap_tokens
                })
            else:
                chunks_with_overlap.append(chunk)
        
        return chunks_with_overlap

    def semantic_merge(self, chunks: List[Dict[str, Any]], similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Merge semantically similar adjacent chunks
        
        Args:
            chunks: List of chunks
            similarity_threshold: Cosine similarity threshold for merging
            
        Returns:
            List of merged chunks
        """
        if similarity_threshold is None:
            similarity_threshold = self.semantic_threshold
        
        if len(chunks) <= 1:
            return chunks
        
        # Create TF-IDF vectors
        texts = [chunk['text'] for chunk in chunks]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        merged_chunks = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i].copy()
            
            # Check if we should merge with next chunk
            if (i + 1 < len(chunks) and 
                similarities[i][i+1] > similarity_threshold and
                current_chunk['tokens'] + chunks[i+1]['tokens'] <= self.target_chunk_size * 1.5):
                
                # Merge chunks
                current_chunk['text'] += '\n\n' + chunks[i+1]['text']
                current_chunk['tokens'] = self.count_tokens(current_chunk['text'])
                current_chunk['split_method'] = 'semantic_merge'
                i += 2  # Skip the next chunk as it's been merged
            else:
                i += 1
            
            merged_chunks.append(current_chunk)
        
        return merged_chunks

    def extract_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from document text
        
        Args:
            text: Document text
            filename: PDF filename
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            'doc_id': filename.replace('.pdf', ''),
            'case_number': '',
            'date': '',
            'judges': [],
            'filename': filename
        }
        
        # Extract case number
        for pattern in self.case_number_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['case_number'] = match.group(0)
                break
        
        # Extract date
        for pattern in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['date'] = match.group(0)
                break
        
        # Extract judges
        for pattern in self.judge_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            metadata['judges'].extend(matches)
        
        # Remove duplicates from judges
        metadata['judges'] = list(set(metadata['judges']))
        
        return metadata

    def build_chunks(self, doc_text: str, blocks: List[Dict[str, Any]], 
                    metadata: Dict[str, Any], footnotes: List[Dict[str, Any]], 
                    words: List[WordInfo]) -> List[Dict[str, Any]]:
        """
        Build final chunks with complete metadata
        
        Args:
            doc_text: Full document text
            blocks: Structural blocks
            metadata: Document metadata
            footnotes: Detected footnotes
            words: Word information for page mapping
            
        Returns:
            List of final chunks with metadata
        """
        all_chunks = []
        chunk_id = 0
        
        for block in blocks:
            # Recursively split the block
            block_chunks = self.recursive_split(block['text'])
            
            # Apply semantic merging
            block_chunks = self.semantic_merge(block_chunks)
            
            # Apply small chunk merging
            block_chunks = self.merge_small_chunks(block_chunks)
            
            # Create chunk metadata
            for i, chunk in enumerate(block_chunks):
                # Find actual page numbers for this chunk
                chunk_pages = self._find_chunk_pages(chunk['text'], words, doc_text)
                
                # Get section classification information
                section_classification = block.get('section_classification')
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
                
                chunk_metadata = ChunkMetadata(
                    doc_id=metadata['doc_id'],
                    case_number=metadata['case_number'],
                    pages=chunk_pages,
                    section=block.get('section_id', ''),
                    section_type=section_type,
                    section_hierarchy=section_hierarchy,
                    footnotes=[],  # Will be populated with relevant footnotes
                    date=metadata['date'],
                    judges=metadata['judges'],
                    char_offset=(0, len(chunk['text'])),  # Simplified for now
                    chunk_id=f"{metadata['doc_id']}_chunk_{chunk_id}",
                    parent_section=block.get('type', 'paragraph'),
                    font_info={},
                    classification_confidence=classification_confidence,
                    classification_evidence=classification_evidence
                )
                
                # Find relevant footnotes for this chunk
                relevant_footnotes = self._find_relevant_footnotes(chunk['text'], footnotes)
                chunk_metadata.footnotes = [fn['text'] for fn in relevant_footnotes]
                
                all_chunks.append({
                    'text': chunk['text'],
                    'metadata': chunk_metadata,
                    'tokens': chunk['tokens'],
                    'split_method': chunk.get('split_method', 'unknown')
                })
                
                chunk_id += 1
        
        return all_chunks

    def _find_chunk_pages(self, chunk_text: str, words: List[WordInfo], full_text: str) -> List[int]:
        """Find the page numbers where this chunk appears"""
        if not chunk_text.strip():
            return [1, 1]
        
        # Find the position of chunk text in full text
        chunk_start = full_text.find(chunk_text[:100])  # Use first 100 chars for matching
        if chunk_start == -1:
            return [1, 1]
        
        # Find corresponding word positions
        char_pos = 0
        start_page = 1
        end_page = 1
        
        for word in words:
            if char_pos >= chunk_start:
                start_page = word.page_num
                break
            char_pos += len(word.text) + 1  # +1 for space
        
        # Find end page
        chunk_end = chunk_start + len(chunk_text)
        char_pos = 0
        
        for word in words:
            if char_pos >= chunk_end:
                end_page = word.page_num
                break
            char_pos += len(word.text) + 1
        
        return [start_page, end_page]

    def _find_relevant_footnotes(self, chunk_text: str, footnotes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find footnotes relevant to a specific chunk"""
        relevant_footnotes = []
        
        # Simple keyword matching for now - use basic word splitting instead of NLTK
        chunk_words = set(chunk_text.lower().split())
        
        for footnote in footnotes:
            try:
                footnote_words = set(footnote['text'].lower().split())
                
                # If there's significant word overlap, consider it relevant
                overlap = len(chunk_words.intersection(footnote_words))
                if overlap > 3:  # At least 3 words in common
                    relevant_footnotes.append(footnote)
            except Exception as e:
                # Skip problematic footnotes
                continue
        
        return relevant_footnotes

    def merge_small_chunks(self, chunks: List[Dict[str, Any]], 
                          min_size: int = None, target_size: int = None, 
                          max_size: int = None) -> List[Dict[str, Any]]:
        """
        Merge adjacent small chunks to reduce fragmentation
        
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
        
        for chunk in chunks:
            tokens = chunk['tokens']
            
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
        
        # Update page range to cover all chunks
        if 'metadata' in base_chunk:
            all_pages = []
            for chunk in chunks:
                if 'metadata' in chunk and hasattr(chunk['metadata'], 'pages'):
                    all_pages.extend(chunk['metadata'].pages)
            
            if all_pages:
                base_chunk['metadata'].pages = [min(all_pages), max(all_pages)]
        
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
        
        # Detect footnotes
        footnotes = self.detect_footnotes(words)
        logger.info(f"Detected {len(footnotes)} footnotes")
        
        # Structural splitting
        blocks = self.structural_split(cleaned_text)
        logger.info(f"Created {len(blocks)} structural blocks")
        
        # Extract metadata
        filename = Path(pdf_path).name
        metadata = self.extract_metadata(cleaned_text, filename)
        
        # Build final chunks
        chunks = self.build_chunks(cleaned_text, blocks, metadata, footnotes, words)
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
                'doc_id': metadata.doc_id,
                'case_number': metadata.case_number,
                'pages': metadata.pages,
                'section': metadata.section,
                'section_type': metadata.section_type,
                'section_hierarchy': metadata.section_hierarchy,
                'footnotes': metadata.footnotes,
                'date': metadata.date,
                'judges': metadata.judges,
                'char_offset': metadata.char_offset,
                'parent_section': metadata.parent_section,
                'classification_confidence': metadata.classification_confidence,
                'classification_evidence': metadata.classification_evidence,
                'tokens': chunk['tokens'],
                'split_method': chunk['split_method']
            })
        
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def main():
    """Main function to run the chunking pipeline"""
    # Initialize chunker
    chunker = ICCJudgmentChunker(target_chunk_size=700, overlap_ratio=0.18)
    
    # Process all PDFs in the past_judgements directory
    input_directory = "/Users/christophe.anglade/Documents/icc_chatbot/data/AI IHL /past_judgements"
    output_path = "/Users/christophe.anglade/Documents/icc_chatbot/output/icc_judgments_chunks.parquet"
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Process all PDFs
    logger.info("Starting PDF processing...")
    all_chunks = chunker.process_directory(input_directory)
    
    # Save to parquet
    chunker.save_to_parquet(all_chunks, output_path)
    
    logger.info(f"Processing complete! Created {len(all_chunks)} chunks from all PDFs.")
    
    # Print sample chunk
    if all_chunks:
        sample_chunk = all_chunks[0]
        print("\nSample chunk:")
        print(f"Text: {sample_chunk['text'][:200]}...")
        print(f"Metadata: {sample_chunk['metadata']}")


if __name__ == "__main__":
    main()
