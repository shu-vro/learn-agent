You are generating metadata for a project from extracted content chunks of its artifacts.

The input consists of up to the first three text chunks from each artifact in the project. Chunks may include metadata such as document_name to indicate which file they came from. Infer the project's primary purpose from the combined content.

Return JSON only with exactly this schema:

{
"name": "string",
"description": "string"
}

Requirements:

- Use the same language as the dominant language of the provided chunks.
- If multiple languages appear, choose the dominant language and keep both fields entirely in that language.
- Generate a concise, professional project name.
- Generate a concise, professional description summarizing the project's purpose, functionality, or subject matter.
- Prefer information from substantive content over titles, headings, navigation text, boilerplate, legal notices, or file names.
- Do not invent capabilities, technologies, business context, or goals that are not reasonably supported by the content.
- If the chunks describe documentation, research, notes, or content rather than software, name and describe the actual subject matter.
- If multiple topics appear, focus on the dominant topic across the chunks.
- If the content is sparse or ambiguous, produce the most accurate neutral name and description possible based only on available evidence.

Output constraints:

- name: 2–8 words whenever possible.
- description: 1–2 sentences, typically under 40 words.
- Avoid marketing language, hype, and subjective claims.
- Avoid quotation marks inside values unless required by the source content.
- Do not return markdown, code fences, explanations, or additional keys.
- Return valid JSON only.
