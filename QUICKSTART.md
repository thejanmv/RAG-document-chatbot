# 🚀 Quick Start Guide

Get up and running with the RAG Document Assistant in 5 minutes!

## Step 1: Install Dependencies (2 min)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Step 2: Get API Keys (2 min)

### Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your key

### Pinecone API Key
1. Visit [Pinecone](https://app.pinecone.io/)
2. Sign up for free account
3. Go to "API Keys" section
4. Copy your API key

## Step 3: Configure Environment (1 min)

```bash
# Copy the example environment file
cp env.example .env

# Edit .env and add your keys
nano .env  # or use any text editor
```

Your `.env` should look like:
```env
GEMINI_API_KEY=AIzaSy...your_actual_key
PINECONE_API_KEY=pcsk_...your_actual_key
PINECONE_INDEX_NAME=rag-docs
```

## Step 4: Verify Setup (30 sec)

```bash
python setup_check.py
```

You should see all green checkmarks ✅

## Step 5: Run the App (30 sec)

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎉 You're Ready!

### Try It Out:

1. **Upload a Document**
   - Click "Upload Documents" tab
   - Choose a PDF, DOCX, or CSV file
   - Click "Process Documents"
   - Wait for success message

2. **Ask Questions**
   - Switch to "Ask Questions" tab
   - Type: "What is this document about?"
   - Click "Ask"
   - Get your AI-powered answer!

## 📝 Example Questions to Try

- "What are the main topics discussed?"
- "Summarize the key findings"
- "What recommendations are provided?"
- "Explain [specific concept] from the document"
- "What are the conclusions?"

## 🐛 Troubleshooting

**App won't start?**
- Check that all dependencies are installed: `pip list`
- Verify Python version: `python --version` (need 3.10+)

**API errors?**
- Double-check your API keys in `.env`
- Ensure no extra spaces or quotes around keys
- Verify keys are valid on respective platforms

**Document processing fails?**
- Make sure file is text-based (not scanned images)
- Try a different file format
- Check file isn't corrupted

## 💡 Tips

- Use **gemini-1.5-flash** for faster responses (default)
- Start with **3 chunks** for retrieval (good balance)
- Check **"Show retrieved context"** to see sources
- Use the **sidebar** to adjust settings

## 🔗 Useful Links

- [Full Documentation](README.md)
- [Google Gemini Docs](https://ai.google.dev/docs)
- [Pinecone Docs](https://docs.pinecone.io/)
- [Streamlit Docs](https://docs.streamlit.io/)

## 🆘 Need Help?

Open an issue on GitHub with:
- Error message (if any)
- Steps you followed
- Python version
- Operating system

---

**Happy querying! 🎯**

