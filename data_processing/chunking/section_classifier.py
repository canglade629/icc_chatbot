"""
Enhanced Section Classification System for Legal Documents

This module provides intelligent section detection and classification for both
ICC judgments and Geneva Convention documents. It uses pattern matching,
keyword analysis, and contextual clues to accurately identify different
types of legal content sections.

Key Features:
- Unified section taxonomy for both document types
- Pattern-based detection with fallback to keyword analysis
- Context-aware classification using surrounding text
- Confidence scoring for classification accuracy
- Support for hierarchical section structures
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SectionType(Enum):
    """Enumeration of all possible section types"""
    # Shared section types
    METADATA = "metadata"
    FACTS_BACKGROUND = "facts_background"
    LEGAL_PROVISION = "legal_provision"
    PROCEDURAL_HISTORY = "procedural_history"
    LEGAL_ANALYSIS = "legal_analysis"
    FINDINGS_JUDGMENT = "findings_judgment"
    OBLIGATIONS_PROHIBITIONS = "obligations_prohibitions"
    DEFINITIONS = "definitions"
    COMMENTARY_INTERPRETATION = "commentary_interpretation"
    ANNEX_REFERENCES = "annex_references"
    
    # ICC-specific section types
    HEADER_METADATA = "header_metadata"
    PROCEDURAL_HISTORY_ICC = "procedural_history_icc"
    FACTUAL_FINDINGS = "factual_findings"
    LEGAL_ISSUES = "legal_issues"
    APPLICABLE_LAW = "applicable_law"
    LEGAL_ANALYSIS_ICC = "legal_analysis_icc"
    FINDINGS_ICC = "findings_icc"
    JUDGMENT_DECISION = "judgment_decision"
    SEPARATE_DISSENTING_OPINIONS = "separate_dissenting_opinions"
    
    # Geneva Convention-specific section types
    TITLE_PREAMBLE = "title_preamble"
    PART_CHAPTER_HEADING = "part_chapter_heading"
    ARTICLE = "article"
    DEFINITION = "definition"
    OBLIGATIONS = "obligations"
    PROHIBITIONS = "prohibitions"
    CONDITIONS_EXCEPTIONS = "conditions_exceptions"
    ENFORCEMENT_IMPLEMENTATION = "enforcement_implementation"
    COMMENTARY_INTERPRETATION_GC = "commentary_interpretation_gc"
    ANNEX_PROTOCOL = "annex_protocol"

@dataclass
class SectionClassification:
    """Result of section classification"""
    section_type: SectionType
    confidence: float
    evidence: List[str]
    context_clues: List[str]
    hierarchical_level: int = 1

class LegalSectionClassifier:
    """
    Main classifier for legal document sections
    """
    
    def __init__(self):
        self._build_patterns()
        self._build_keywords()
        self._build_context_rules()
    
    def _build_patterns(self):
        """Build regex patterns for section detection"""
        
        # ICC Judgment patterns
        self.icc_patterns = {
            SectionType.HEADER_METADATA: [
                r'^Case\s+No\.?\s*ICC-\d+/\d+-\d+',
                r'^International\s+Criminal\s+Court',
                r'^Chamber\s+[IVXLC]+',
                r'^Judge\s+[A-Z][a-z]+',
                r'^Presiding\s+Judge',
                r'^Registry\s+of\s+the\s+Court',
                r'^Date\s*:?\s*\d{1,2}\s+\w+\s+\d{4}',
                r'^Filed\s+on\s+\d{1,2}\s+\w+\s+\d{4}',
            ],
            
            SectionType.PROCEDURAL_HISTORY_ICC: [
                r'^Procedural\s+History',
                r'^Background\s+and\s+Procedural\s+History',
                r'^Procedural\s+Background',
                r'^History\s+of\s+the\s+Proceedings',
                r'^Procedural\s+Context',
            ],
            
            SectionType.FACTUAL_FINDINGS: [
                r'^Factual\s+Findings',
                r'^Facts\s+and\s+Circumstances',
                r'^Factual\s+Background',
                r'^The\s+Facts',
                r'^Factual\s+Context',
                r'^Background\s+Facts',
            ],
            
            SectionType.LEGAL_ISSUES: [
                r'^Legal\s+Issues',
                r'^Issues\s+for\s+Determination',
                r'^Questions\s+of\s+Law',
                r'^Legal\s+Questions',
                r'^Issues\s+in\s+Dispute',
            ],
            
            SectionType.APPLICABLE_LAW: [
                r'^Applicable\s+Law',
                r'^Relevant\s+Law',
                r'^Legal\s+Framework',
                r'^Statutory\s+Provisions',
                r'^Legal\s+Provisions',
                r'^Relevant\s+Provisions',
            ],
            
            SectionType.LEGAL_ANALYSIS_ICC: [
                r'^Legal\s+Analysis',
                r'^Analysis\s+of\s+the\s+Law',
                r'^Legal\s+Reasoning',
                r'^Analysis\s+and\s+Reasoning',
                r'^Legal\s+Discussion',
            ],
            
            SectionType.FINDINGS_ICC: [
                r'^Findings',
                r'^Court\s+Findings',
                r'^Determination',
                r'^Conclusions',
                r'^Court\s+Conclusions',
            ],
            
            SectionType.JUDGMENT_DECISION: [
                r'^JUDGMENT\s*$',
                r'^DECISION\s*$',
                r'^ORDER\s*$',
                r'^RULING\s*$',
                r'^Final\s+Judgment',
                r'^Court\s+Decision',
            ],
            
            SectionType.SEPARATE_DISSENTING_OPINIONS: [
                r'^Separate\s+Opinion',
                r'^Dissenting\s+Opinion',
                r'^Concurring\s+Opinion',
                r'^Individual\s+Opinion',
                r'^Opinion\s+of\s+Judge',
            ],
        }
        
        # Geneva Convention patterns
        self.geneva_patterns = {
            SectionType.TITLE_PREAMBLE: [
                r'^Geneva\s+Convention\s+[IVXLC]+',
                r'^Protocol\s+Additional\s+[IVXLC]+',
                r'^Rome\s+Statute',
                r'^Preamble',
                r'^Whereas',
                r'^Considering\s+that',
            ],
            
            SectionType.PART_CHAPTER_HEADING: [
                r'^Part\s+([IVXLC]+)',
                r'^PART\s+([IVXLC]+)',
                r'^Chapter\s+([IVXLC]+)',
                r'^CHAPTER\s+([IVXLC]+)',
                r'^Title\s+([IVXLC]+)',
            ],
            
            SectionType.ARTICLE: [
                r'^Article\s+(\d+[a-z]?)\s*[:\.]?\s*(.*)',
                r'^Art\.\s+(\d+[a-z]?)\s*[:\.]?\s*(.*)',
                r'^ARTICLE\s+(\d+[a-z]?)\s*[:\.]?\s*(.*)',
            ],
            
            SectionType.DEFINITION: [
                r'^Definition\s+of\s+',
                r'^For\s+the\s+purposes\s+of\s+this\s+',
                r'^In\s+this\s+[Cc]onvention',
                r'^The\s+term\s+"[^"]+"\s+means',
                r'^As\s+used\s+in\s+this\s+',
            ],
            
            SectionType.OBLIGATIONS: [
                r'^Obligations\s+of\s+',
                r'^States\s+Parties\s+shall',
                r'^Each\s+State\s+Party\s+shall',
                r'^The\s+[Cc]onvention\s+requires',
                r'^shall\s+ensure\s+that',
                r'^undertakes\s+to',
            ],
            
            SectionType.PROHIBITIONS: [
                r'^Prohibitions',
                r'^shall\s+not\s+',
                r'^is\s+prohibited',
                r'^are\s+prohibited',
                r'^No\s+person\s+shall',
                r'^It\s+is\s+prohibited',
            ],
            
            SectionType.CONDITIONS_EXCEPTIONS: [
                r'^Conditions\s+and\s+Exceptions',
                r'^Exception\s+to',
                r'^Subject\s+to\s+the\s+following',
                r'^Unless\s+otherwise\s+provided',
                r'^In\s+exceptional\s+circumstances',
            ],
            
            SectionType.ENFORCEMENT_IMPLEMENTATION: [
                r'^Enforcement\s+and\s+Implementation',
                r'^Implementation\s+of\s+the\s+',
                r'^Compliance\s+with\s+this\s+',
                r'^Monitoring\s+of\s+compliance',
                r'^Oversight\s+mechanisms',
            ],
            
            SectionType.COMMENTARY_INTERPRETATION_GC: [
                r'^Commentary',
                r'^Comment\s+on\s+Article',
                r'^Interpretation\s+of\s+Article',
                r'^Explanatory\s+Note',
                r'^Clarification',
            ],
            
            SectionType.ANNEX_PROTOCOL: [
                r'^Annex\s+[IVXLC\d]+',
                r'^ANNEX\s+[IVXLC\d]+',
                r'^Appendix\s+[IVXLC\d]+',
                r'^Protocol\s+[IVXLC\d]+',
                r'^Schedule\s+[IVXLC\d]+',
            ],
        }
        
        # Shared patterns
        self.shared_patterns = {
            SectionType.METADATA: [
                r'^Document\s+Information',
                r'^Case\s+Information',
                r'^Document\s+Details',
                r'^File\s+Reference',
            ],
            
            SectionType.FACTS_BACKGROUND: [
                r'^Background',
                r'^Facts',
                r'^Circumstances',
                r'^Context',
                r'^History',
            ],
            
            SectionType.LEGAL_PROVISION: [
                r'^Legal\s+Provision',
                r'^Statutory\s+Provision',
                r'^Article\s+\d+',
                r'^Section\s+\d+',
            ],
            
            SectionType.PROCEDURAL_HISTORY: [
                r'^Procedural\s+History',
                r'^Procedural\s+Background',
                r'^History\s+of\s+Proceedings',
            ],
            
            SectionType.LEGAL_ANALYSIS: [
                r'^Analysis',
                r'^Reasoning',
                r'^Discussion',
                r'^Legal\s+Analysis',
            ],
            
            SectionType.FINDINGS_JUDGMENT: [
                r'^Findings',
                r'^Judgment',
                r'^Decision',
                r'^Conclusion',
            ],
            
            SectionType.OBLIGATIONS_PROHIBITIONS: [
                r'^Obligations',
                r'^Prohibitions',
                r'^Duties',
                r'^Rights\s+and\s+Obligations',
            ],
            
            SectionType.DEFINITIONS: [
                r'^Definitions',
                r'^Definition\s+of\s+Terms',
                r'^Glossary',
            ],
            
            SectionType.COMMENTARY_INTERPRETATION: [
                r'^Commentary',
                r'^Interpretation',
                r'^Explanatory\s+Notes',
            ],
            
            SectionType.ANNEX_REFERENCES: [
                r'^Annex',
                r'^Appendix',
                r'^References',
                r'^Bibliography',
            ],
        }
    
    def _build_keywords(self):
        """Build keyword dictionaries for section classification"""
        
        # ICC-specific keywords
        self.icc_keywords = {
            SectionType.HEADER_METADATA: [
                'case number', 'chamber', 'judge', 'presiding', 'registry',
                'date', 'filed', 'international criminal court', 'icc',
                'prosecutor', 'defendant', 'accused', 'victim'
            ],
            
            SectionType.PROCEDURAL_HISTORY_ICC: [
                'procedural history', 'background', 'proceedings', 'motions',
                'submissions', 'rulings', 'procedural context', 'case history',
                'trial proceedings', 'preliminary proceedings'
            ],
            
            SectionType.FACTUAL_FINDINGS: [
                'factual findings', 'facts', 'circumstances', 'events',
                'chronology', 'timeline', 'evidence', 'witness testimony',
                'factual background', 'case narrative'
            ],
            
            SectionType.LEGAL_ISSUES: [
                'legal issues', 'questions', 'determination', 'dispute',
                'legal questions', 'issues for determination', 'matters',
                'points of law', 'legal matters'
            ],
            
            SectionType.APPLICABLE_LAW: [
                'applicable law', 'relevant law', 'legal framework',
                'statutory provisions', 'rome statute', 'rules of procedure',
                'geneva conventions', 'international law', 'customary law'
            ],
            
            SectionType.LEGAL_ANALYSIS_ICC: [
                'legal analysis', 'reasoning', 'analysis', 'discussion',
                'legal reasoning', 'interpretation', 'application of law',
                'legal discussion', 'court analysis'
            ],
            
            SectionType.FINDINGS_ICC: [
                'findings', 'determination', 'conclusions', 'court findings',
                'conclusion', 'decision', 'ruling', 'verdict'
            ],
            
            SectionType.JUDGMENT_DECISION: [
                'judgment', 'decision', 'order', 'ruling', 'final judgment',
                'court decision', 'disposition', 'verdict'
            ],
            
            SectionType.SEPARATE_DISSENTING_OPINIONS: [
                'separate opinion', 'dissenting opinion', 'concurring opinion',
                'individual opinion', 'opinion of judge', 'dissenting',
                'concurring', 'separate'
            ],
        }
        
        # Geneva Convention keywords
        self.geneva_keywords = {
            SectionType.TITLE_PREAMBLE: [
                'geneva convention', 'protocol additional', 'rome statute',
                'preamble', 'whereas', 'considering', 'treaty', 'convention'
            ],
            
            SectionType.PART_CHAPTER_HEADING: [
                'part', 'chapter', 'title', 'section', 'division'
            ],
            
            SectionType.ARTICLE: [
                'article', 'art.', 'provision', 'clause'
            ],
            
            SectionType.DEFINITION: [
                'definition', 'means', 'refers to', 'for the purposes',
                'in this convention', 'term', 'as used in'
            ],
            
            SectionType.OBLIGATIONS: [
                'obligations', 'shall', 'must', 'undertakes to', 'required to',
                'duty', 'responsibility', 'states parties shall'
            ],
            
            SectionType.PROHIBITIONS: [
                'prohibited', 'shall not', 'must not', 'forbidden',
                'prohibition', 'ban', 'not permitted'
            ],
            
            SectionType.CONDITIONS_EXCEPTIONS: [
                'conditions', 'exceptions', 'subject to', 'unless',
                'except', 'exception', 'circumstances'
            ],
            
            SectionType.ENFORCEMENT_IMPLEMENTATION: [
                'enforcement', 'implementation', 'compliance', 'monitoring',
                'oversight', 'supervision', 'control'
            ],
            
            SectionType.COMMENTARY_INTERPRETATION_GC: [
                'commentary', 'comment', 'interpretation', 'explanation',
                'clarification', 'note', 'explanation'
            ],
            
            SectionType.ANNEX_PROTOCOL: [
                'annex', 'appendix', 'protocol', 'schedule', 'attachment'
            ],
        }
        
        # Shared keywords
        self.shared_keywords = {
            SectionType.METADATA: [
                'document', 'information', 'details', 'reference', 'file',
                'case', 'document information'
            ],
            
            SectionType.FACTS_BACKGROUND: [
                'background', 'facts', 'circumstances', 'context', 'history',
                'events', 'situation'
            ],
            
            SectionType.LEGAL_PROVISION: [
                'legal provision', 'statutory provision', 'article', 'section',
                'clause', 'paragraph'
            ],
            
            SectionType.PROCEDURAL_HISTORY: [
                'procedural history', 'procedural background', 'proceedings',
                'history of proceedings', 'procedural context'
            ],
            
            SectionType.LEGAL_ANALYSIS: [
                'analysis', 'reasoning', 'discussion', 'legal analysis',
                'legal reasoning', 'interpretation'
            ],
            
            SectionType.FINDINGS_JUDGMENT: [
                'findings', 'judgment', 'decision', 'conclusion', 'ruling',
                'determination', 'verdict'
            ],
            
            SectionType.OBLIGATIONS_PROHIBITIONS: [
                'obligations', 'prohibitions', 'duties', 'rights and obligations',
                'requirements', 'restrictions'
            ],
            
            SectionType.DEFINITIONS: [
                'definitions', 'definition of terms', 'glossary', 'terms',
                'meaning', 'defines'
            ],
            
            SectionType.COMMENTARY_INTERPRETATION: [
                'commentary', 'interpretation', 'explanatory notes', 'comment',
                'explanation', 'clarification'
            ],
            
            SectionType.ANNEX_REFERENCES: [
                'annex', 'appendix', 'references', 'bibliography', 'sources',
                'attachments'
            ],
        }
    
    def _build_context_rules(self):
        """Build context-based classification rules"""
        
        self.context_rules = {
            # Rules for ICC judgments
            'icc': {
                'after_header': [SectionType.PROCEDURAL_HISTORY_ICC, SectionType.FACTUAL_FINDINGS],
                'after_facts': [SectionType.LEGAL_ISSUES, SectionType.APPLICABLE_LAW],
                'after_law': [SectionType.LEGAL_ANALYSIS_ICC],
                'after_analysis': [SectionType.FINDINGS_ICC, SectionType.JUDGMENT_DECISION],
                'before_opinions': [SectionType.SEPARATE_DISSENTING_OPINIONS],
            },
            
            # Rules for Geneva Conventions
            'geneva': {
                'after_preamble': [SectionType.PART_CHAPTER_HEADING, SectionType.ARTICLE],
                'after_part': [SectionType.ARTICLE],
                'after_article': [SectionType.DEFINITION, SectionType.OBLIGATIONS, SectionType.PROHIBITIONS],
                'in_article': [SectionType.DEFINITION, SectionType.OBLIGATIONS, SectionType.PROHIBITIONS, SectionType.CONDITIONS_EXCEPTIONS],
            }
        }
    
    def classify_section(self, text: str, document_type: str, 
                        context: Optional[Dict[str, Any]] = None) -> SectionClassification:
        """
        Classify a text section based on content and context
        
        Args:
            text: The text to classify
            document_type: 'icc' or 'geneva'
            context: Optional context information (previous sections, etc.)
            
        Returns:
            SectionClassification object
        """
        text_lower = text.lower().strip()
        
        # Try pattern matching first
        pattern_result = self._classify_by_patterns(text, document_type)
        if pattern_result.confidence > 0.8:
            return pattern_result
        
        # Try keyword matching
        keyword_result = self._classify_by_keywords(text, document_type)
        if keyword_result.confidence > 0.6:
            return keyword_result
        
        # Try context-based classification
        if context:
            context_result = self._classify_by_context(text, document_type, context)
            if context_result.confidence > 0.4:
                return context_result
        
        # Default to generic classification
        return self._classify_generic(text, document_type)
    
    def _classify_by_patterns(self, text: str, document_type: str) -> SectionClassification:
        """Classify using regex patterns"""
        text_lower = text.lower().strip()
        best_match = None
        best_confidence = 0.0
        evidence = []
        
        # Get appropriate patterns based on document type
        patterns = {}
        if document_type == 'icc':
            patterns.update(self.icc_patterns)
        elif document_type == 'geneva':
            patterns.update(self.geneva_patterns)
        patterns.update(self.shared_patterns)
        
        for section_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    confidence = 0.9  # High confidence for pattern matches
                    if confidence > best_confidence:
                        best_match = section_type
                        best_confidence = confidence
                        evidence.append(f"Pattern match: {pattern}")
        
        return SectionClassification(
            section_type=best_match or SectionType.METADATA,
            confidence=best_confidence,
            evidence=evidence,
            context_clues=[]
        )
    
    def _classify_by_keywords(self, text: str, document_type: str) -> SectionClassification:
        """Classify using keyword analysis"""
        text_lower = text.lower()
        best_match = None
        best_score = 0.0
        evidence = []
        
        # Get appropriate keywords based on document type
        keywords = {}
        if document_type == 'icc':
            keywords.update(self.icc_keywords)
        elif document_type == 'geneva':
            keywords.update(self.geneva_keywords)
        keywords.update(self.shared_keywords)
        
        for section_type, keyword_list in keywords.items():
            score = 0
            matched_keywords = []
            
            for keyword in keyword_list:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            if score > best_score:
                best_match = section_type
                best_score = score
                evidence = matched_keywords
        
        # Calculate confidence based on keyword density
        confidence = min(0.8, best_score / 3.0) if best_score > 0 else 0.0
        
        return SectionClassification(
            section_type=best_match or SectionType.METADATA,
            confidence=confidence,
            evidence=evidence,
            context_clues=[]
        )
    
    def _classify_by_context(self, text: str, document_type: str, 
                           context: Dict[str, Any]) -> SectionClassification:
        """Classify using context rules"""
        if document_type not in self.context_rules:
            return SectionClassification(
                section_type=SectionType.METADATA,
                confidence=0.0,
                evidence=[],
                context_clues=[]
            )
        
        rules = self.context_rules[document_type]
        previous_section = context.get('previous_section_type')
        
        if previous_section and previous_section in rules:
            possible_types = rules[previous_section]
            # For now, return the first possible type with medium confidence
            return SectionClassification(
                section_type=possible_types[0],
                confidence=0.5,
                evidence=[],
                context_clues=[f"Context: follows {previous_section}"]
            )
        
        return SectionClassification(
            section_type=SectionType.METADATA,
            confidence=0.0,
            evidence=[],
            context_clues=[]
        )
    
    def _classify_generic(self, text: str, document_type: str) -> SectionClassification:
        """Generic classification when other methods fail"""
        text_lower = text.lower().strip()
        
        # Simple heuristics
        if len(text) < 50:
            return SectionClassification(
                section_type=SectionType.METADATA,
                confidence=0.3,
                evidence=["Short text"],
                context_clues=["Generic classification"]
            )
        elif any(word in text_lower for word in ['article', 'section', 'paragraph']):
            return SectionClassification(
                section_type=SectionType.LEGAL_PROVISION,
                confidence=0.4,
                evidence=["Contains legal structure words"],
                context_clues=["Generic classification"]
            )
        else:
            return SectionClassification(
                section_type=SectionType.FACTS_BACKGROUND,
                confidence=0.3,
                evidence=["Default classification"],
                context_clues=["Generic classification"]
            )
    
    def get_section_hierarchy(self, section_type: SectionType) -> int:
        """Get the hierarchical level of a section type"""
        hierarchy_levels = {
            # Level 1 (highest)
            SectionType.TITLE_PREAMBLE: 1,
            SectionType.HEADER_METADATA: 1,
            SectionType.PART_CHAPTER_HEADING: 1,
            
            # Level 2
            SectionType.ARTICLE: 2,
            SectionType.PROCEDURAL_HISTORY_ICC: 2,
            SectionType.FACTUAL_FINDINGS: 2,
            SectionType.LEGAL_ISSUES: 2,
            SectionType.APPLICABLE_LAW: 2,
            
            # Level 3
            SectionType.LEGAL_ANALYSIS_ICC: 3,
            SectionType.LEGAL_ANALYSIS: 3,
            SectionType.FINDINGS_ICC: 3,
            SectionType.FINDINGS_JUDGMENT: 3,
            
            # Level 4
            SectionType.JUDGMENT_DECISION: 4,
            SectionType.SEPARATE_DISSENTING_OPINIONS: 4,
            SectionType.OBLIGATIONS: 4,
            SectionType.PROHIBITIONS: 4,
            SectionType.CONDITIONS_EXCEPTIONS: 4,
            
            # Level 5 (lowest)
            SectionType.DEFINITION: 5,
            SectionType.DEFINITIONS: 5,
            SectionType.COMMENTARY_INTERPRETATION: 5,
            SectionType.COMMENTARY_INTERPRETATION_GC: 5,
            SectionType.ANNEX_REFERENCES: 5,
            SectionType.ANNEX_PROTOCOL: 5,
        }
        
        return hierarchy_levels.get(section_type, 3)
    
    def is_icc_section(self, section_type: SectionType) -> bool:
        """Check if a section type is specific to ICC judgments"""
        icc_sections = {
            SectionType.HEADER_METADATA,
            SectionType.PROCEDURAL_HISTORY_ICC,
            SectionType.FACTUAL_FINDINGS,
            SectionType.LEGAL_ISSUES,
            SectionType.APPLICABLE_LAW,
            SectionType.LEGAL_ANALYSIS_ICC,
            SectionType.FINDINGS_ICC,
            SectionType.JUDGMENT_DECISION,
            SectionType.SEPARATE_DISSENTING_OPINIONS,
        }
        return section_type in icc_sections
    
    def is_geneva_section(self, section_type: SectionType) -> bool:
        """Check if a section type is specific to Geneva Conventions"""
        geneva_sections = {
            SectionType.TITLE_PREAMBLE,
            SectionType.PART_CHAPTER_HEADING,
            SectionType.ARTICLE,
            SectionType.DEFINITION,
            SectionType.OBLIGATIONS,
            SectionType.PROHIBITIONS,
            SectionType.CONDITIONS_EXCEPTIONS,
            SectionType.ENFORCEMENT_IMPLEMENTATION,
            SectionType.COMMENTARY_INTERPRETATION_GC,
            SectionType.ANNEX_PROTOCOL,
        }
        return section_type in geneva_sections
