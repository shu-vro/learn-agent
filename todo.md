1. [x] check if qdrant is working.
2. [x] migrate from faiss to qdrant.
3. [x] if a paper is already uploaded, skip the ingestion process and directly use the existing index for retrieval. not true for --rebuild
   - the way this will work is:
     1. it will download the paper
     2. hash it in sha256, save in each point's metadata. this will be the unique identifier for each paper.
     3. check if the hash exists in qdrant. if it does, skip ingestion. if not, ingest and for now, print it.
4. [x] add support for multiple paper ingestion.
5. [x] introduce a stable database: eg postgres to store hash.
6. [ ] make stable ui with web page
   - [x] use nextjs for frontend.
     - [x] react-mosaic for tiling
   - [x] use fastapi for backend.

7. [x] artifact ingestion is a long running task, so we need to introduce a queue system to handle it.
8. [x] update project name and description based on the artifacts uploaded.
       here's the algorithm:
   1. if a project doesn't have a name and description,
   - on the artifact upload, it will get first 3 chunks of each artifacts uploaded, feed it to llm and generate a name and description for the project.
   - [x] add a side-agent helper that turns `Document[]` into project name/description JSON.
   - [x] write the prompt so the model returns strict JSON in the same language as the chunks.
   2. if a project already has a name and description, nothing happens.

9. [x] add support for uploading and handling multiple artifacts in a single project on a single upload
10. [ ] add model picker preset with reasoning settings. also add this in user's settings for preferences. user can change default model and it will load next time user goes to /chat.
