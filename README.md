# RAG LLM Slim Blueprint

This blueprint allows is a one click to deploy a RAG Slim pipeline for inference using LLM connected to cnvrg storage solution.

## Prerequisite

1. A Large Language Model hosted on cnvrg, OpenAI or HuggingFace.
2. A cnvrg dataset holding the relevant documents to be used for RAG endpoint. The dataset needs to be added to the flow as a data task.
3. In order to keep the FastRAG endpoint up-to-date with newly added data use the [continual learning](https://app.cnvrg.io/docs/core_concepts/flows.html#settings) feature in the flow configurations. For every file change of the conected dataset a new version of the FastRAG endpoint will be launched with access to the latest files.

## Flow
1. Click on `Use Blueprint` button.
2. You will be redirected to a new project with the blueprint flow page.
3. Go to the project settings section and update the [environment variables](https://app.cnvrg.io/docs/core_concepts/projects.html#environment) with relevant information that will be used by the RAG endpoint. 
For more info see the component [documentation](https://app.af2jdjq262tdqvyelihtqnd.cloud.cnvrg.io/blueprintsdev/blueprints/libraries/rag-endpoint-slim/1.0.0)
4. Link the cnvrg dataset as a task with the inference.
5. Click on continual learning and select `Trigger on dataset update` and choose your dataset
6. Click on the ‘Run Flow’ button
7. In a few minutes you will have a RAG endpoint
8. Go to the ‘Serving’ tab in the project and look for your endpoint.
9. You can use the Try it Live section to query the RAG endpoint and generate relevant answers with LLM connected.
10. You can also integrate your API with your code using the integration panel at the bottom of the page
