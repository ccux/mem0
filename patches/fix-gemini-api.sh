#!/bin/bash
# Fix Gemini API compatibility issues in mem0 package
# This script updates the mem0 package to work with the current Google Generative AI API

set -e

echo "=== Applying Gemini API fixes to mem0 package ==="

# Fix imports in gemini.py embeddings
sed -i 's/from google import genai/import google.generativeai as genai/' /opt/venv/lib/python3.11/site-packages/mem0/embeddings/gemini.py
sed -i 's/from google.genai import types/from google.generativeai import types/' /opt/venv/lib/python3.11/site-packages/mem0/embeddings/gemini.py

# Fix client initialization
sed -i 's/self.client = genai.Client(api_key=api_key)/genai.configure(api_key=api_key)/' /opt/venv/lib/python3.11/site-packages/mem0/embeddings/gemini.py

# Fix embed_content method call
sed -i 's/response = self.client.models.embed_content(model=self.config.model, contents=text, config=config)/response = genai.embed_content(model=self.config.model, content=text, output_dimensionality=self.config.embedding_dims)/' /opt/venv/lib/python3.11/site-packages/mem0/embeddings/gemini.py

# Fix response access
sed -i 's/return response.embeddings\[0\].values/return response["embedding"]/' /opt/venv/lib/python3.11/site-packages/mem0/embeddings/gemini.py

# Remove obsolete config line
sed -i '/config = types.EmbedContentConfig/d' /opt/venv/lib/python3.11/site-packages/mem0/embeddings/gemini.py

echo "=== Gemini API fixes applied successfully ==="
