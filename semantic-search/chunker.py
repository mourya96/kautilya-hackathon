"""
Chunking module for creating intelligent documentation chunks.
Organizes API endpoint documentation into searchable chunks with metadata.
"""

from typing import Dict, List, Any


class DocumentationChunker:
    """Creates intelligent chunks from API endpoint documentation."""

    def __init__(self, max_chunk_tokens: int = 512):
        """
        Initialize the chunker.

        Args:
            max_chunk_tokens: Maximum tokens per chunk (approximate)
        """
        self.max_chunk_tokens = max_chunk_tokens

    def create_chunks(self, endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create searchable chunks from endpoint documentation.

        Args:
            endpoints: List of endpoint dictionaries from data_loader

        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []

        for endpoint in endpoints:
            # Create a comprehensive text representation of the endpoint
            chunk_text = self._format_endpoint(endpoint)

            # Create metadata for filtering and context
            metadata = {
                'endpoint_name': endpoint['name'],
                'category': endpoint['category'],
                'method': endpoint['method'],
                'url': endpoint['url'],
            }

            chunks.append({
                'text': chunk_text,
                'metadata': metadata
            })

        return chunks

    def _format_endpoint(self, endpoint: Dict[str, Any]) -> str:
        """
        Format an endpoint into a comprehensive text representation.

        Args:
            endpoint: Endpoint dictionary

        Returns:
            Formatted text string
        """
        parts = []

        # Add endpoint name and category
        parts.append(f"Endpoint: {endpoint['name']}")
        parts.append(f"Category: {endpoint['category']}")
        parts.append(f"Method: {endpoint['method']}")
        parts.append(f"URL: {endpoint['url']}")

        # Add description
        if endpoint['description']:
            parts.append(f"\nDescription: {endpoint['description']}")

        # Add path variables
        if endpoint['path_variables']:
            parts.append("\nPath Variables:")
            for var in endpoint['path_variables']:
                var_desc = var.get('description', 'No description')
                parts.append(f"  - {var['key']} ({var.get('type', 'string')}): {var_desc}")

        # Add query parameters
        if endpoint['parameters']:
            parts.append("\nQuery Parameters:")
            for param in endpoint['parameters']:
                if not param.get('disabled', False):
                    param_desc = param.get('description', 'No description')
                    parts.append(f"  - {param['key']}: {param_desc}")

        # Add response examples
        if endpoint['response_examples']:
            parts.append("\nResponse Examples:")
            for resp in endpoint['response_examples']:
                parts.append(f"  - {resp['name']} (HTTP {resp.get('code', 'N/A')})")

        return '\n'.join(parts)

    def split_large_chunks(self, chunks: List[Dict[str, Any]],
                          max_chars: int = 2000) -> List[Dict[str, Any]]:
        """
        Split chunks that are too large into smaller pieces.

        Args:
            chunks: List of chunk dictionaries
            max_chars: Maximum characters per chunk

        Returns:
            List of chunks with large ones split
        """
        result = []

        for chunk in chunks:
            text = chunk['text']

            if len(text) <= max_chars:
                result.append(chunk)
            else:
                # Split by sections (separated by double newlines)
                sections = text.split('\n\n')
                current_chunk = []
                current_length = 0

                for section in sections:
                    section_length = len(section)

                    if current_length + section_length > max_chars and current_chunk:
                        # Save current chunk
                        result.append({
                            'text': '\n\n'.join(current_chunk),
                            'metadata': chunk['metadata'].copy()
                        })
                        current_chunk = [section]
                        current_length = section_length
                    else:
                        current_chunk.append(section)
                        current_length += section_length + 2  # +2 for \n\n

                # Add remaining chunk
                if current_chunk:
                    result.append({
                        'text': '\n\n'.join(current_chunk),
                        'metadata': chunk['metadata'].copy()
                    })

        return result


def create_documentation_chunks(endpoints: List[Dict[str, Any]],
                                max_chunk_tokens: int = 512) -> List[Dict[str, Any]]:
    """
    Convenience function to create documentation chunks.

    Args:
        endpoints: List of endpoint dictionaries
        max_chunk_tokens: Maximum tokens per chunk

    Returns:
        List of chunk dictionaries
    """
    chunker = DocumentationChunker(max_chunk_tokens)
    chunks = chunker.create_chunks(endpoints)
    chunks = chunker.split_large_chunks(chunks)
    return chunks
