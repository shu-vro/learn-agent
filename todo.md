1. [x] check if qdrant is working.
2. [x] migrate from faiss to qdrant.
3. [x] if a paper is already uploaded, skip the ingestion process and directly use the existing index for retrieval. not true for --rebuild
   - the way this will work is:
     1. it will download the paper
     2. hash it in sha256, save in each point's metadata. this will be the unique identifier for each paper.
     3. check if the hash exists in qdrant. if it does, skip ingestion. if not, ingest and for now, print it.
4. [x] add support for multiple paper ingestion.
5. [x] introduce a stable database: eg postgres to store hash.
6. [ ] integrate mem0 with rag_agent.
7. [ ] make stable ui with web page
   - [x] use nextjs for frontend.
     - [x] react-mosaic for tiling
   - [x] use fastapi for backend.

8. [x] artifact ingestion is a long running task, so we need to introduce a queue system to handle it.
