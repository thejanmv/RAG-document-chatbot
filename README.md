# 📚 RAG-Powered Document Assistant (OpenAI Version)

An AI-powered document chatbot that enables users to upload documents (PDF, DOCX, CSV), indexes their content into a vector database, and provides intelligent question-answering using Retrieval-Augmented Generation (RAG) powered by OpenAI GPT models.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 Features

- **Multi-Format Support**: Upload and process PDF, DOCX, and CSV files
- **OpenAI Integration**: Uses text-embedding-3-small for embeddings and GPT-4o-mini/GPT-4o for generation
- **Vector Search**: Semantic search powered by Pinecone vector database
- **Interactive UI**: Clean and intuitive Streamlit web interface
- **Real-time Processing**: Upload documents and get answers instantly
- **Context Display**: View retrieved document chunks that support each answer
- **Model Selection**: Choose between GPT-4o-mini (fast), GPT-4o (capable), GPT-3.5-turbo, or GPT-4-turbo

## 🏗️ Architecture

```
┌─────────────┐
│   Upload    │
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Text Extraction    │
│  (PyPDF2, docx2txt) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Text Chunking      │
│  (LangChain)        │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  OpenAI Embeddings  │
│ (text-embedding-3-  │
│      small)         │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Pinecone Storage   │
│  (Vector DB)        │
└─────────────────────┘

Query Flow:
User Question → OpenAI Embedding → Pinecone Search → 
Retrieved Context → OpenAI GPT → Generated Answer
```

## 📋 Prerequisites

- Python 3.10 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Pinecone API key ([Get one here](https://app.pinecone.io/))

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG-document-chatbot
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Index Name (optional - defaults to "rag-docs")
PINECONE_INDEX_NAME=rag-docs
```

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload Documents**
   - Navigate to the "Upload Documents" tab
   - Click "Choose files" and select your PDF, DOCX, or CSV files
   - Click "Process Documents" to index them
   - Wait for processing to complete (you'll see a success message)

2. **Ask Questions**
   - Switch to the "Ask Questions" tab
   - Type your question in the input box
   - Click "Ask" to get an AI-generated answer
   - Toggle "Show retrieved context" to see supporting document chunks

3. **Adjust Settings**
   - Use the sidebar to select the OpenAI model (GPT-4o-mini, GPT-4o, etc.)
   - Adjust the number of chunks to retrieve (1-10)
   - View index statistics to see how many documents are stored

## 📁 Project Structure

```
rag-document-chatbot/
│
├── app.py                  # Streamlit web interface
├── ingest.py              # Document processing and embedding
├── query_engine.py        # Retrieval and answer generation
├── requirements.txt       # Python dependencies
├── env.example           # Environment variables template
├── README.md             # This file
│
├── data/                 # Sample documents directory
│   └── README.md
│
└── .env                  # Your API keys (not in git)
```

## 🔧 Configuration

### Chunking Parameters

Modify in `ingest.py`:

```python
CHUNK_SIZE = 1000      # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
```

### Retrieval Parameters

Modify in `query_engine.py`:

```python
TOP_K = 3  # Number of chunks to retrieve
```

### Model Selection

Available OpenAI models:
- `gpt-4o-mini` - Fast & cheap (default) - Best value
- `gpt-4o` - Most capable, highest quality
- `gpt-3.5-turbo` - Cheapest option
- `gpt-4-turbo` - Previous generation, good balance

## 🧪 Testing

### Test Document Ingestion

```python
python -c "from ingest import DocumentProcessor; print('Ingest module loaded successfully')"
```

### Test Query Engine

```python
from query_engine import test_query_engine
test_query_engine("What are the main topics?")
```

## 📊 Technical Details

### Embeddings
- **Model**: `text-embedding-3-small` (OpenAI)
- **Dimension**: 1536
- **Cost**: $0.02 per 1M tokens

### Language Models
- **Default**: `gpt-4o-mini` - $0.15 input / $0.60 output per 1M tokens
- **Premium**: `gpt-4o` - $2.50 input / $10.00 output per 1M tokens
- **Budget**: `gpt-3.5-turbo` - $0.50 input / $1.50 output per 1M tokens
- **Context**: Top-K retrieved chunks with metadata

### Vector Database
- **Provider**: Pinecone
- **Metric**: Cosine similarity
- **Index Type**: Serverless (AWS us-east-1)
- **Dimension**: 1536 (for text-embedding-3-small)

## 🐛 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: Invalid API key
```
- Ensure your `.env` file has valid API keys
- Check that `.env` is in the project root directory
- Verify keys start with `sk-` for OpenAI and `pcsk_` for Pinecone

**2. Pinecone Connection Issues**
```
Error connecting to Pinecone index
```
- Verify your Pinecone API key is correct
- Check if the index exists in your Pinecone dashboard
- The app will create the index automatically if it doesn't exist (serverless on AWS us-east-1)

**3. Document Processing Errors**
```
Error extracting text from PDF
```
- Ensure the PDF is text-based (not scanned images)
- Try a different PDF reader if issues persist

**4. Import Errors**
```
ModuleNotFoundError: No module named 'X'
```
- Reinstall dependencies: `pip install -r requirements.txt`
- Ensure virtual environment is activated

**5. Rate Limit Errors**
```
Rate limit exceeded
```
- OpenAI has rate limits on the API
- Wait a moment and try again
- Consider upgrading your OpenAI plan

### Getting Help

If you encounter issues:
1. Check the error message in the Streamlit interface
2. Look at the terminal output for detailed logs
3. Verify all environment variables are set correctly
4. Ensure you have internet connectivity for API calls
5. Check your OpenAI account has sufficient credits

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secrets in Streamlit Cloud settings:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME` (optional)

### Local Production

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📈 Performance Considerations

### OpenAI API Limits

**Rate Limits** (varies by account tier):
- Free tier: 3 requests/minute
- Tier 1: 500 requests/minute
- Tier 2: 5,000 requests/minute

**Token Limits**:
- Free tier: 200,000 tokens/day
- Higher tiers: Much higher limits

Monitor usage at: https://platform.openai.com/usage

### Optimization Tips
1. Use `gpt-4o-mini` for faster and cheaper responses
2. Reduce `TOP_K` value for quicker retrieval
3. Decrease `CHUNK_SIZE` for more granular search
4. Process documents in batches for large uploads
5. Cache frequently asked questions

### Cost Optimization

**Embeddings**: text-embedding-3-small
- $0.02 per 1M tokens
- ~750 words per 1,000 tokens
- Example: 100-page document ≈ $0.05

**Generation**: gpt-4o-mini (default)
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens
- Example: 10 questions ≈ $0.01

**Total typical cost**: $0.10 - $1.00 per day for moderate use

## 🔐 Security Best Practices

1. **Never commit `.env` file** to version control
2. **Rotate API keys** regularly
3. **Use environment variables** for all secrets
4. **Validate file uploads** before processing
5. **Implement rate limiting** in production
6. **Monitor API usage** to prevent unexpected charges
7. **Set spending limits** in OpenAI dashboard

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for embeddings and language generation
- **Pinecone** for vector database
- **LangChain** for document processing utilities
- **Streamlit** for the web interface framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🗺️ Roadmap

Future enhancements:
- [ ] Multi-language support
- [ ] Conversation history persistence
- [ ] Export chat transcripts
- [ ] Advanced filtering and search options
- [ ] Support for more document formats (TXT, MD, HTML)
- [ ] Batch processing for large document sets
- [ ] Citation tracking and source highlighting
- [ ] Custom prompt templates
- [ ] Evaluation metrics dashboard
- [ ] Streaming responses
- [ ] Function calling support
- [ ] Multi-modal support (images in PDFs)

## 💡 Tips & Best Practices

### For Best Results:

1. **Document Quality**
   - Use clear, well-formatted documents
   - Avoid scanned PDFs (use OCR first)
   - Keep documents under 10MB each

2. **Question Quality**
   - Ask specific, focused questions
   - Reference specific topics in documents
   - Use follow-up questions for clarity

3. **Model Selection**
   - Use `gpt-4o-mini` for most queries (fast & cheap)
   - Use `gpt-4o` for complex analysis
   - Use `gpt-3.5-turbo` for simple lookups

4. **Chunk Settings**
   - More chunks = More context but slower
   - Start with 3 chunks, adjust as needed
   - Monitor response quality vs speed

## 📊 Comparison: OpenAI vs Gemini

| Feature | OpenAI (This Version) | Gemini (Previous) |
|---------|----------------------|-------------------|
| **Embedding Model** | text-embedding-3-small | models/embedding-001 |
| **Embedding Dimension** | 1536 | 768 |
| **LLM Model** | gpt-4o-mini / gpt-4o | gemini-1.5-flash/pro |
| **Pricing** | Pay-as-you-go | Free tier (limited) |
| **Rate Limits** | Higher (paid) | 1,500 embeds/day (free) |
| **Context Window** | 128K tokens | 32K - 1M tokens |
| **Quality** | Excellent | Excellent |
| **Speed** | Very fast | Very fast |

**Why OpenAI?**
- No free tier quota limits that you hit
- More predictable costs
- Better documentation
- Wider model selection
- Industry standard

---

**Built with ❤️ using OpenAI GPT**
