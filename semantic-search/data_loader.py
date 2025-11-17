"""
Data loader module for parsing Postman collection JSON files.
Extracts API endpoint documentation from Twitter API v2 Postman collection.
"""

import json
from typing import Dict, List, Any
from pathlib import Path


class PostmanCollectionLoader:
    """Loads and parses Postman collection JSON files."""

    def __init__(self, collection_path: str):
        """
        Initialize the loader with a path to a Postman collection.

        Args:
            collection_path: Path to the Postman collection JSON file
        """
        self.collection_path = Path(collection_path)
        self.collection_data = None

    def load(self) -> Dict[str, Any]:
        """Load the Postman collection JSON file."""
        with open(self.collection_path, 'r', encoding='utf-8') as f:
            self.collection_data = json.load(f)
        return self.collection_data

    def extract_endpoints(self) -> List[Dict[str, Any]]:
        """
        Extract all API endpoints from the Postman collection.

        Returns:
            List of endpoint dictionaries with structured information
        """
        if not self.collection_data:
            self.load()

        endpoints = []
        self._extract_items(self.collection_data.get('item', []), endpoints, [])
        return endpoints

    def _extract_items(self, items: List[Dict], endpoints: List[Dict], path: List[str]):
        """
        Recursively extract endpoints from nested Postman collection items.

        Args:
            items: List of Postman collection items
            endpoints: List to accumulate extracted endpoints
            path: Current path in the collection hierarchy (for categorization)
        """
        for item in items:
            # If item has nested items (folder), recurse
            if 'item' in item:
                new_path = path + [item.get('name', 'Unknown')]
                self._extract_items(item['item'], endpoints, new_path)
            # If item has a request, it's an endpoint
            elif 'request' in item:
                endpoint = self._parse_endpoint(item, path)
                endpoints.append(endpoint)

    def _parse_endpoint(self, item: Dict, category_path: List[str]) -> Dict[str, Any]:
        """
        Parse a single endpoint item into a structured format.

        Args:
            item: Postman collection item representing an endpoint
            category_path: Category hierarchy for this endpoint

        Returns:
            Dictionary with endpoint information
        """
        request = item.get('request', {})

        # Extract basic info
        name = item.get('name', 'Unnamed Endpoint')
        description = item.get('description', request.get('description', ''))
        method = request.get('method', 'GET')

        # Extract URL information
        url_info = request.get('url', {})
        if isinstance(url_info, str):
            url = url_info
            path_parts = []
            query_params = []
        else:
            # Build URL from components
            protocol = url_info.get('protocol', 'https')
            host = '.'.join(url_info.get('host', []))
            path_parts = url_info.get('path', [])
            url = f"{protocol}://{host}/{'/'.join(path_parts)}"
            query_params = url_info.get('query', [])

        # Extract path variables
        path_variables = url_info.get('variable', []) if isinstance(url_info, dict) else []

        # Parse query parameters
        parameters = []
        for param in query_params:
            param_info = {
                'key': param.get('key', ''),
                'description': param.get('description', ''),
                'disabled': param.get('disabled', False)
            }
            parameters.append(param_info)

        # Parse path variables
        variables = []
        for var in path_variables:
            var_info = {
                'key': var.get('key', ''),
                'description': var.get('description', ''),
                'type': var.get('type', 'string')
            }
            variables.append(var_info)

        # Extract response examples
        responses = []
        for response in item.get('response', []):
            resp_info = {
                'name': response.get('name', ''),
                'code': response.get('code', 200),
                'status': response.get('status', '')
            }
            responses.append(resp_info)

        return {
            'name': name,
            'category': ' > '.join(category_path) if category_path else 'General',
            'method': method,
            'url': url,
            'description': description,
            'parameters': parameters,
            'path_variables': variables,
            'response_examples': responses
        }


def load_twitter_api_docs(collection_path: str) -> List[Dict[str, Any]]:
    """
    Convenience function to load Twitter API documentation.

    Args:
        collection_path: Path to the Postman collection JSON file

    Returns:
        List of endpoint dictionaries
    """
    loader = PostmanCollectionLoader(collection_path)
    return loader.extract_endpoints()
