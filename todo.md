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
10. [x] add model picker preset with reasoning settings. also add this in user's settings for preferences. user can change default model and it will load next time user goes to /chat.
11. [ ] problem: if a service goes down, everything goes down. for example, if qdrant goes down, users can't even see their other threads.
12. [ ] a feature: it will process each chunk of the document in llm and generate notes on it. save it in vector db again, with also in the chunk section in the database.
        process:
    - when user uploads a document, it generally makes chunks out of it. after that happens, i want that a background job will run by taking consecutive 3 chunks together using sliding window, pass them in llm and generating notes based on those 3 chunks, where the 2nd one will be the main one and the target note will be about the 2nd one. then we will save this note with all the other chunks, with the type of NOTE_CHUNK_TYPE. users will not wait for notes. when notes will be generated, user can see them. beside every chunk, there will be a note button. when clicked, the note will be shown. make a settings section which will imply user wants to generate notes out of chunks of documents. if yes, the background job will run in the background and user can continue with their work. make sure that concurrency is used to generate notes for all the chunks. at most 7 concurrent jobs will run. the number 7 here will come from src.config.constants.MAX_CONCURRENT_NOTES_GENERATION.

    what that note taking process include:
    1. introduction of that chunk
    2. description
    3. summary
    4. analytical questions based on those chunks section if any.
    5. those question's answer. don't hesitate to make the notes broad.

    the note generator llm call will happen in a side-agent. so we need to create a side-agent for this.

    make sure to also generate the ui for this. what will happen is on each chunk, there will be a button on the top, like a accordion saying: notes. when clicked, the note will be shown. make sure to also generate the ui for this.

    also, if chunks are basically images and texts, make sure to generate the notes for the images as well.
    remember, question is an optional yet important part. if generated, make sure they are hard as university final exams or phd levels. which will really let the user understand the content better.
