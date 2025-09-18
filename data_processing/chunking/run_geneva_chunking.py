#!/usr/bin/env python3
"""
Geneva Convention PDF Chunking Pipeline Runner

This script runs the specialized PDF chunking pipeline for Geneva Convention
documents, supplementary protocols, statutes, and other law articles.
"""

import sys
import os
from pathlib import Path

# Add the chunking module to the path
sys.path.append(str(Path(__file__).parent / "chunking"))

from geneva_convention_chunker import GenevaConventionChunker, main

if __name__ == "__main__":
    print("Geneva Convention PDF Chunking Pipeline")
    print("=" * 50)
    
    # Run the main chunking pipeline
    main()
