"""
Chunking service for splitting documents into semantic chunks.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent splitting.
"""

import logging
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    chunk_index: int
    total_chunks: int
    section_title: Optional[str]
    char_count: int


class ChunkingService:
    """
    Handles intelligent text chunking with context preservation.
    Prioritizes document structure (paragraphs > sentences > words).
    """
    
    def __init__(self):
        settings = get_settings()
        
        # Configure splitter with structural separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=[
                "\n\n",      # Paragraphs (highest priority)
                "\n",        # Lines
                ". ",        # Sentences
                "? ",        # Questions
                "! ",        # Exclamations
                "; ",        # Clauses
                ", ",        # Phrases
                " ",         # Words
                ""           # Characters (last resort)
            ],
            length_function=len,
            is_separator_regex=False,
        )
        
        # Patterns for detecting section titles
        self.heading_patterns = [
            r'^#{1,6}\s+(.+)$',           # Markdown headings
            r'^\[Page \d+\]$',             # Page markers
            r'^\[Slide \d+\]$',            # Slide markers
            r'^[A-Z][A-Z\s]{5,50}$',       # ALL CAPS headings (5-50 chars)
            r'^\d+\.\s+[A-Z]',             # Numbered sections starting with caps
            r'^Chapter\s+\d+',             # Chapter headings
            r'^Section\s+\d+',             # Section headings
        ]
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Full document text
            
        Returns:
            List of TextChunk objects with metadata
        """
        if not text or not text.strip():
            return []
        
        # Extract sections with titles
        sections = self._extract_sections(text)
        
        all_chunks = []
        chunk_index = 0
        
        for section_title, section_text in sections:
            # Split this section into chunks
            if section_text.strip():
                section_chunks = self.splitter.split_text(section_text)
                
                for chunk_content in section_chunks:
                    if chunk_content.strip():
                        all_chunks.append(TextChunk(
                            content=chunk_content.strip(),
                            chunk_index=chunk_index,
                            total_chunks=0,  # Will be updated after
                            section_title=section_title,
                            char_count=len(chunk_content.strip())
                        ))
                        chunk_index += 1
        
        # Update total_chunks for all chunks
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total
        
        logger.info(f"Created {total} chunks from {len(text)} characters")
        return all_chunks
    
    def _extract_sections(self, text: str) -> List[Tuple[Optional[str], str]]:
        """
        Extract sections with their titles from the text.
        
        Returns:
            List of (section_title, section_content) tuples
        """
        lines = text.split('\n')
        sections = []
        current_title = None
        current_content = []
        
        for line in lines:
            # Check if this line is a heading
            heading = self._detect_heading(line)
            
            if heading:
                # Save previous section if exists
                if current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((current_title, content))
                
                current_title = heading
                current_content = []
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((current_title, content))
        
        # If no sections were found, treat entire text as one section
        if not sections and text.strip():
            sections.append((None, text.strip()))
        
        return sections
    
    def _detect_heading(self, line: str) -> Optional[str]:
        """
        Detect if a line is a heading/title.
        
        Returns:
            The heading text if detected, None otherwise
        """
        line = line.strip()
        if not line:
            return None
        
        for pattern in self.heading_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                # Return the captured group if exists, otherwise the whole match
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a text.
        Rough approximation: 1 token ≈ 4 characters for English text.
        """
        return len(text) // 4


# Singleton instance
chunking_service = ChunkingService()
