import { TextLoader } from "langchain/document_loaders/fs/text";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { FaissStore } from "langchain/vectorstores/faiss";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import {
  JSONLinesLoader,
  JSONLoader,
} from "langchain/document_loaders/fs/json";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAI } from "langchain/llms/openai";
import { RetrievalQAChain } from "langchain/chains";
import express, { Application, Request, Response } from "express";

/** Langchain / Open AI */
// Set up the AI model
const model = new OpenAI({});

// Load docs
const loader = new DirectoryLoader("src/data", {
  ".json": (path) => new JSONLoader(path, "/texts"),
  ".jsonl": (path) => new JSONLinesLoader(path, "/html"),
  ".txt": (path) => new TextLoader(path),
  ".csv": (path) => new CSVLoader(path, "text"),
});

const docs = await loader.load();

// Split docs
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 50,
  separators: ["\r\n", "\n", " "],
});

const splitDocs = await splitter.splitDocuments(docs);

// Create the vector store
const vectorStore = await FaissStore.fromDocuments(
  splitDocs,
  new OpenAIEmbeddings()
);

// Save the vector store
const directory = "src/embeddings";

await vectorStore.save(directory);

// Set up the chain
const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

/** Api  */
// Create the express Api
const app: Application = express();
const port = 5018;

// Body parsing Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// The end point
app.post(
  "/chat",
  async (req: any, res: Response): Promise<Response> => {
    const answer = await chain.call({
      query: req.body.question,
    });
    return res.status(200).send({
      message: answer,
    });
  }
);

// Start the Api
try {
  app.listen(port, (): void => {
    console.log(`Connected successfully on port ${port}`);
  });
} catch (error: any) {
  console.error(`Error occured: ${error.message}`);
}
