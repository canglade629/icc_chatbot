#!/usr/bin/env python3
"""
ICC Judgment PDF Chunking Pipeline Runner with Progress Logging

This script runs the complete PDF chunking pipeline on all ICC judgment PDFs
in the past_judgements directory and outputs the results as a parquet file.
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add the chunking module to the path
sys.path.append(str(Path(__file__).parent / "chunking"))

from pdf_chunker import ICCJudgmentChunker

def main():
    """Main function to run the chunking pipeline with progress logging"""
    print("=" * 80)
    print("ICC Judgment PDF Chunking Pipeline")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize chunker
    chunker = ICCJudgmentChunker(target_chunk_size=700, overlap_ratio=0.18)
    
    # Process all PDFs in the past_judgements directory
    input_directory = "/Users/christophe.anglade/Documents/icc_chatbot/data/AI IHL /past_judgements"
    output_path = "/Users/christophe.anglade/Documents/icc_chatbot/output/icc_judgments_chunks.parquet"
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = list(Path(input_directory).glob("*.pdf"))
    total_files = len(pdf_files)
    
    print(f"Found {total_files} PDF files to process")
    print(f"Output will be saved to: {output_path}")
    print()
    
    all_chunks = []
    processed_files = 0
    start_time = time.time()
    
    for i, pdf_file in enumerate(pdf_files, 1):
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
        
        # Save intermediate results every 5 files
        if i % 5 == 0:
            print(f"💾 Saving intermediate results... ({len(all_chunks)} chunks)")
            chunker.save_to_parquet(all_chunks, f"{output_path}.intermediate")
            print(f"✓ Intermediate results saved")
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
    main()
