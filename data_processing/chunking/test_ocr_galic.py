#!/usr/bin/env python3
"""
Test OCR functionality on Galic AJ.pdf
"""

import sys
import os
sys.path.append('chunking')

from pdf_chunker import ICCJudgmentChunker

def test_galic_ocr():
    """Test OCR on Galic AJ document"""
    print("Testing OCR on Galic AJ.pdf")
    print("=" * 50)
    
    # Initialize chunker
    chunker = ICCJudgmentChunker()
    pdf_path = 'data/AI IHL /past_judgements/Galic AJ.pdf'
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    print(f"Processing: {pdf_path}")
    
    # Test regular extraction
    print("\n1. Regular text extraction:")
    try:
        text, words = chunker.extract_text_and_words(pdf_path)
        meaningful_text = text.strip()
        print(f"   Characters: {len(meaningful_text)}")
        print(f"   Alphabetic: {len([c for c in meaningful_text if c.isalpha()])}")
        print(f"   Preview: {meaningful_text[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test OCR fallback
    print("\n2. OCR fallback:")
    try:
        text_ocr, words_ocr = chunker.extract_text_with_fallback(pdf_path)
        meaningful_text_ocr = text_ocr.strip()
        print(f"   Characters: {len(meaningful_text_ocr)}")
        print(f"   Alphabetic: {len([c for c in meaningful_text_ocr if c.isalpha()])}")
        print(f"   Preview: {meaningful_text_ocr[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test full processing
    print("\n3. Full processing:")
    try:
        chunks = chunker.process_pdf(pdf_path)
        print(f"   Chunks created: {len(chunks)}")
        if len(chunks) > 0:
            print(f"   Sample chunk: {chunks[0]['text'][:100]}...")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_galic_ocr()
