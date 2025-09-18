#!/usr/bin/env python3
"""
Test script for enhanced chunking with improved section detection

This script demonstrates the new section classification capabilities
for both ICC judgments and Geneva Convention documents.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add the chunking module to the path
sys.path.append(str(Path(__file__).parent / "chunking"))

from section_classifier import LegalSectionClassifier, SectionType
from pdf_chunker import ICCJudgmentChunker
from geneva_convention_chunker import GenevaConventionChunker

def test_section_classifier():
    """Test the section classifier with sample texts"""
    print("=" * 80)
    print("TESTING ENHANCED SECTION CLASSIFIER")
    print("=" * 80)
    
    classifier = LegalSectionClassifier()
    
    # Test ICC judgment samples
    icc_samples = [
        ("Case No. ICC-01/04-01/06", "icc"),
        ("Procedural History\nThis case involves...", "icc"),
        ("Factual Findings\nThe evidence shows that...", "icc"),
        ("Legal Analysis\nThe Court must consider...", "icc"),
        ("JUDGMENT\nThe Court finds the accused guilty...", "icc"),
        ("Separate Opinion of Judge Smith", "icc"),
    ]
    
    print("\nICC Judgment Section Classification:")
    print("-" * 50)
    for text, doc_type in icc_samples:
        classification = classifier.classify_section(text, doc_type)
        print(f"Text: {text[:50]}...")
        print(f"Type: {classification.section_type.value}")
        print(f"Confidence: {classification.confidence:.2f}")
        print(f"Evidence: {classification.evidence}")
        print()
    
    # Test Geneva Convention samples
    geneva_samples = [
        ("Geneva Convention IV", "geneva"),
        ("Article 3 common to the Geneva Conventions", "geneva"),
        ("Article 27: Women shall be especially protected...", "geneva"),
        ("For the purposes of this Convention, the term 'civilian' means...", "geneva"),
        ("States Parties shall ensure that...", "geneva"),
        ("It is prohibited to attack civilians...", "geneva"),
        ("Commentary on Article 3", "geneva"),
        ("Annex I: List of Medical Equipment", "geneva"),
    ]
    
    print("\nGeneva Convention Section Classification:")
    print("-" * 50)
    for text, doc_type in geneva_samples:
        classification = classifier.classify_section(text, doc_type)
        print(f"Text: {text[:50]}...")
        print(f"Type: {classification.section_type.value}")
        print(f"Confidence: {classification.confidence:.2f}")
        print(f"Evidence: {classification.evidence}")
        print()

def test_icc_chunking():
    """Test ICC judgment chunking with enhanced metadata"""
    print("=" * 80)
    print("TESTING ICC JUDGMENT CHUNKING")
    print("=" * 80)
    
    # Find a sample ICC judgment PDF
    icc_dir = Path("/Users/christophe.anglade/Documents/icc_chatbot/data/AI IHL /past_judgements")
    icc_pdfs = list(icc_dir.glob("*.pdf"))
    
    if not icc_pdfs:
        print("No ICC judgment PDFs found for testing")
        return
    
    # Test with the first PDF
    test_pdf = icc_pdfs[0]
    print(f"Testing with: {test_pdf.name}")
    
    chunker = ICCJudgmentChunker(target_chunk_size=500, overlap_ratio=0.15)
    
    try:
        chunks = chunker.process_pdf(str(test_pdf))
        print(f"Successfully processed {len(chunks)} chunks")
        
        # Show sample chunks with enhanced metadata
        print("\nSample chunks with enhanced metadata:")
        print("-" * 50)
        for i, chunk in enumerate(chunks[:3]):
            metadata = chunk['metadata']
            print(f"\nChunk {i+1}:")
            print(f"  Text: {chunk['text'][:100]}...")
            print(f"  Section Type: {metadata.section_type}")
            print(f"  Section Hierarchy: {metadata.section_hierarchy}")
            print(f"  Classification Confidence: {metadata.classification_confidence:.2f}")
            print(f"  Classification Evidence: {metadata.classification_evidence}")
            print(f"  Tokens: {chunk['tokens']}")
            print(f"  Split Method: {chunk['split_method']}")
        
        # Save test results
        output_path = "/Users/christophe.anglade/Documents/icc_chatbot/output/test_icc_enhanced.parquet"
        chunker.save_to_parquet(chunks, output_path)
        print(f"\nTest results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing ICC judgment: {e}")

def test_geneva_chunking():
    """Test Geneva Convention chunking with enhanced metadata"""
    print("=" * 80)
    print("TESTING GENEVA CONVENTION CHUNKING")
    print("=" * 80)
    
    # Find a sample Geneva Convention PDF
    geneva_dir = Path("/Users/christophe.anglade/Documents/icc_chatbot/data/AI IHL /documentation")
    geneva_pdfs = list(geneva_dir.glob("*.pdf"))
    
    if not geneva_pdfs:
        print("No Geneva Convention PDFs found for testing")
        return
    
    # Test with the first PDF
    test_pdf = geneva_pdfs[0]
    print(f"Testing with: {test_pdf.name}")
    
    chunker = GenevaConventionChunker(target_chunk_size=400, overlap_ratio=0.12)
    
    try:
        chunks = chunker.process_pdf(str(test_pdf))
        print(f"Successfully processed {len(chunks)} chunks")
        
        # Show sample chunks with enhanced metadata
        print("\nSample chunks with enhanced metadata:")
        print("-" * 50)
        for i, chunk in enumerate(chunks[:3]):
            metadata = chunk['metadata']
            print(f"\nChunk {i+1}:")
            print(f"  Text: {chunk['text'][:100]}...")
            print(f"  Section Type: {metadata.section_type}")
            print(f"  Section Hierarchy: {metadata.section_hierarchy}")
            print(f"  Classification Confidence: {metadata.classification_confidence:.2f}")
            print(f"  Classification Evidence: {metadata.classification_evidence}")
            print(f"  Article: {metadata.article}")
            print(f"  Part: {metadata.part}")
            print(f"  Tokens: {chunk['tokens']}")
            print(f"  Split Method: {chunk['split_method']}")
        
        # Save test results
        output_path = "/Users/christophe.anglade/Documents/icc_chatbot/output/test_geneva_enhanced.parquet"
        chunker.save_to_parquet(chunks, output_path)
        print(f"\nTest results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing Geneva Convention: {e}")

def analyze_enhanced_metadata():
    """Analyze the enhanced metadata from test results"""
    print("=" * 80)
    print("ANALYZING ENHANCED METADATA")
    print("=" * 80)
    
    # Check if test files exist
    icc_test_file = Path("/Users/christophe.anglade/Documents/icc_chatbot/output/test_icc_enhanced.parquet")
    geneva_test_file = Path("/Users/christophe.anglade/Documents/icc_chatbot/output/test_geneva_enhanced.parquet")
    
    if icc_test_file.exists():
        print("\nICC Judgment Enhanced Metadata Analysis:")
        print("-" * 50)
        df_icc = pd.read_parquet(icc_test_file)
        print(f"Total chunks: {len(df_icc)}")
        print(f"Unique section types: {df_icc['section_type'].nunique()}")
        print(f"Section type distribution:")
        print(df_icc['section_type'].value_counts())
        print(f"Average classification confidence: {df_icc['classification_confidence'].mean():.2f}")
        print(f"High confidence chunks (>0.7): {(df_icc['classification_confidence'] > 0.7).sum()}")
    
    if geneva_test_file.exists():
        print("\nGeneva Convention Enhanced Metadata Analysis:")
        print("-" * 50)
        df_geneva = pd.read_parquet(geneva_test_file)
        print(f"Total chunks: {len(df_geneva)}")
        print(f"Unique section types: {df_geneva['section_type'].nunique()}")
        print(f"Section type distribution:")
        print(df_geneva['section_type'].value_counts())
        print(f"Average classification confidence: {df_geneva['classification_confidence'].mean():.2f}")
        print(f"High confidence chunks (>0.7): {(df_geneva['classification_confidence'] > 0.7).sum()}")

def main():
    """Main test function"""
    print("Enhanced Chunking Test Suite")
    print("=" * 80)
    
    # Ensure output directory exists
    Path("/Users/christophe.anglade/Documents/icc_chatbot/output").mkdir(exist_ok=True)
    
    # Run tests
    test_section_classifier()
    test_icc_chunking()
    test_geneva_chunking()
    analyze_enhanced_metadata()
    
    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
