# LangChain Documentation Helper


This is a simple web application made with RAG for building an AsyncAPI assistant using FAISS as a vectorstore. It answers questions about AsyncAPI based on LangChain's official documentation.



https://github.com/user-attachments/assets/87f13cea-3753-449b-9742-6a07cbb69435




## Tech Stack
Client: Streamlit

Vectorstore: FAISS 

## Environment Variables

To run this project, you will need to add the `OPENAI_API_KEY` environment variables to your .env file


## Run Locally

Clone the project

```bash
  git clone https://github.com/ManishSharma1609/assistant_asyncapi.git
```

Go to the project directory

```bash
  cd assistant_asyncapi
```


Install dependencies

```bash
  pip install -r requirements.txt
```

Start the flask server

```bash
  streamlit run main.py
```
