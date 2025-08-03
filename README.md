# Personal AI Assistant Backend

A sophisticated RAG-powered chatbot backend that serves as an intelligent personal assistant, capable of answering questions about professional background, skills, and experience using retrieval-augmented generation.

## 🎯 Project Overview

This project implements a production-ready AI chatbot backend designed for personal websites and portfolios. It leverages modern AI technologies to create an interactive assistant that can intelligently discuss professional qualifications, work experience, and technical expertise by processing uploaded documents and generating contextually relevant responses.

## ✨ Key Features

- **🤖 Advanced AI Integration**: Utilizes Hugging Face's open-source language models for natural conversation
- **🔍 Intelligent Document Retrieval**: RAG architecture ensures responses are grounded in actual resume content
- **⚡ High-Performance API**: Built with FastAPI for optimal speed and scalability
- **📄 Multi-Format Support**: Processes PDF, DOCX, and TXT documents seamlessly
- **🌐 Web-Ready**: CORS-enabled for easy frontend integration
- **💾 Efficient Search**: FAISS vector database for lightning-fast similarity search
- **🔧 Production-Ready**: Comprehensive error handling and logging

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Document      │
│   (React/Vue)   │◄──►│   Backend       │◄──►│   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   RAG Engine    │◄──►│   Vector Store  │
                       │   (LangChain)   │    │   (FAISS)       │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Hugging Face   │
                       │     Models      │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for model loading)
- Internet connection (for initial model download)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-chatbot-backend.git
   cd ai-chatbot-backend
   ```

2. **Set up virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

5. **Add your documents**
   - Place your resume, CV, or other relevant documents in the `documents/` folder
   - Supported formats: PDF, DOCX, TXT

6. **Start the server**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status and information |
| `GET` | `/health` | Health check for monitoring |
| `POST` | `/chat` | Main chat interface |
| `GET` | `/documents` | List processed documents |
| `GET` | `/docs` | Interactive API documentation |

### Chat Endpoint

**POST** `/chat`

```json
{
  "message": "What experience do you have with Python?",
  "user_id": "optional-user-identifier"
}
```

**Response:**
```json
{
  "response": "I have extensive experience with Python...",
  "sources": ["resume.pdf", "portfolio.txt"],
  "confidence": 0.85
}
```

## 🔧 Configuration

Key environment variables in `.env`:

```env
# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHAT_MODEL=distilgpt2

# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=https://yourwebsite.com

# Storage Configuration
VECTOR_STORE_PATH=./vector_store
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🌐 Frontend Integration

### React Example
```javascript
const chatWithAssistant = async (message) => {
  try {
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    
    const data = await response.json();
    return data.response;
  } catch (error) {
    console.error('Chat error:', error);
  }
};
```

### Vanilla JavaScript
```javascript
async function askQuestion(question) {
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: question })
  });
  return await response.json();
}
```

## 🧪 Testing

Test the API using curl:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "Tell me about your technical skills"}'
```

## 📈 Performance & Scalability

- **Model Loading**: ~30-60 seconds initial startup
- **Response Time**: ~1-3 seconds per query
- **Memory Usage**: ~2-4GB with loaded models
- **Concurrent Users**: Supports multiple simultaneous requests
- **Document Limit**: No hard limit, performance scales with document count

## 🛠️ Development

### Project Structure
```
├── main.py              # FastAPI application entry point
├── models.py            # Pydantic data models
├── chatbot.py           # Core chatbot logic
├── rag_system.py        # RAG implementation
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── documents/           # Document storage
└── vector_store/        # Generated embeddings
```

### Adding Features
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]
```

### Cloud Platforms
- **Railway**: One-click deployment from GitHub
- **Render**: Automatic builds and deployments
- **Google Cloud Run**: Serverless container deployment
- **AWS ECS**: Enterprise-grade container orchestration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Live Demo**: [Your deployed app URL]
- **Portfolio**: [Your portfolio website]
- **LinkedIn**: [Your LinkedIn profile]

## 💡 Future Enhancements

- [ ] Multi-language support
- [ ] Voice interaction capabilities
- [ ] Advanced conversation memory
- [ ] Custom model fine-tuning
- [ ] Analytics and usage tracking
- [ ] WebSocket support for real-time chat

---

**Built with ❤️ for creating intelligent, personalized web experiences.**
