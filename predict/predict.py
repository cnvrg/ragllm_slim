import os
from haystack.pipelines import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, PromptNode, PromptModel
from fastrag.rankers.colbert import ColBERTRanker
import http.client
import json
from haystack.nodes import BM25Retriever, SentenceTransformersRanker, PromptTemplate
from transformers import AutoTokenizer
import requests
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document

class main_endpoint:
    def __init__(self):
        self.cnvrg = True
        
    def read_environ_variables(self):
        # pipeline varirables
        # print(os.environ)
        self.provider = os.environ["PROVIDER"]

        self.dataset_name = self.check_variable("DATASET")
        self.retrieverk = int(self.check_variable("RETRIEVER_N"))
        self.rankerk = int(self.check_variable("RANKER_N"))
        self.model_name = self.check_variable("MODEL_NAME")
        
        # llm variables
        self.api_key = os.environ["API_KEY"]
        self.document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)

        if self.provider.lower() != "cnvrg":
            self.cnvrg = False
            
        # Azure support will be added in the next release
        # self.deployment_name = self.check_variable("AZURE_DEPLOYMENT_NAME")
        # self.base_url = self.check_variable("AZURE_BASE_URL")

        # setup cnvrg credentials
        
        if self.provider == "cnvrg":
            self.cnvrg_url = os.environ["URL"]
            self.cnvrg_1 = self.cnvrg_url[
                len("https://") : self.cnvrg_url.rfind("cnvrg.io") + len("cnvrg.io")
            ]
            self.cnvrg_2 = self.cnvrg_url[
                self.cnvrg_url.rfind("cnvrg.io") + len("cnvrg.io") :
            ]

        # define the prompt text
        self.prompt_text = self.check_variable("PROMPT")

    def check_variable(self, variable):
        try:
            return os.environ[variable]
        except KeyError:
            return None
        
    def updator(self, document_name):
        data = json.load(open(document_name, "r"))
        contents = [
            Document(
                content="A patient asked: "
                + d["input"]
                + ". The doctor answered: "
                + d["output"]
            )
            for d in data
        ]
        self.document_store.write_documents(contents)
        return "updated"
    
    def RAG_pipeline(self):
        
        for document in os.listdir(f"/data/{self.dataset_name}/"):
            if document.endswith('.json'):
                self.updator(f"/data/{self.dataset_name}/"+document)

        retriever = BM25Retriever(
            document_store=self.document_store, top_k=int(self.retrieverk)
        )
        ranker = ColBERTRanker(checkpoint_path="Intel/ColBERT-NQ", top_k=int(self.rankerk))
        # ranker = SentenceTransformersRanker(
        #     model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2",
        #     top_k=5,
        #     batch_size=32,
        #     use_gpu=False,
        # )
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=ranker, name="Reranker", inputs=["Retriever"])

    def huggingface_query(self, text):
        
        # if not working hardcode tokens to 750
        encoded = self.tk.encode(text)
        # limited = text[:750]
        limited = encoded[:self.tk.model_max_length]
        decoded = self.tk.decode(limited, skip_special_tokens=True)

        output = {
            "inputs": decoded,
            "parameters": {"max_new_tokens": self.tk.model_max_length},
        }
        response = requests.post(self.API_URL, headers=self.headers, json=output)
        return response.json()

    def external_language_model(self):

        if self.provider.lower() == "openai":
            # model = PromptModel(self.model_name, api_key=self.api_key)
            self.LLM = PromptNode(self.model_name, api_key=self.api_key)
            self.cnvrg = False

        elif self.provider.lower() == "huggingface":
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.api_key

            self.tk = AutoTokenizer.from_pretrained(self.model_name)
            self.API_URL = (
                f"https://api-inference.huggingface.co/models/{self.model_name}"
            )
            self.headers = {"Authorization": f"Bearer {self.api_key}"}

            self.LLM = self.huggingface_query

        else:
            raise Exception(
                "Please provide a valid LLM service provider in the environment variable PROVIDER, acceptable ones are cnvrg, openai, huggingface"
            )

        # Azure support will be added in the next release
        # if self.deployment_name is not None:
        #     model = PromptModel(
        #         model_name_or_path=self.model_name,
        #         api_key=self.api_key,
        #         model_kwargs={
        #             "azure_deployment_name": self.deployment_name,
        #             "azure_base_url": self.base_url,
        #         },
        #     )
        #     self.LLM = PromptNode(model)
        
    def cnvrg_language_model(self, data):

        conn = http.client.HTTPSConnection(self.cnvrg_1, 443)
        headers = {"Cnvrg-Api-Key": self.api_key, "Content-Type": "application/json"}
        request_dict = {"prompt": data}
        payload = '{"input_params":' + json.dumps(request_dict) + "}"

        conn.request("POST", self.cnvrg_2, payload, headers)

        res = conn.getresponse()
        data = res.read()

        return data.decode("utf-8")

    def call_llm(self, data):

        if self.cnvrg:
            return self.cnvrg_language_model(data)
        else:
            return self.LLM(data)

# os.environ["PROVIDER"] = 'cnvrg'
# os.environ["DATASET"] = 'rag'
# os.environ["RETRIEVER_N"] = '10'
# os.environ["RANKER_N"] = '5'
# os.environ["MODEL_NAME"] = 'google/flan-t5-xxl'
# os.environ["API_KEY"] = 'SXj1CvN18Jg35Wh8yQEVJb1V'
# os.environ["URL"] = 'https://inference-1031-1.amr2uxdpwunywjqvp2kefkp.cloud.cnvrg.io/api/v1/endpoints/sdchqsvw1kfb4nsn29wk'
# os.environ["PROMPT"] = '''Below is an instruction that describes a task paired with an input, which provides further context. Write a response that appropriately completes the request.

#     ### Instruction:
#     You are a doctor. Synthesize a comprehensive answer from the following Input and the question: {query}

#     ### Input:
#     paragraphs: {documents}

#     ### Response:'''

definitions = main_endpoint()
definitions.read_environ_variables()
definitions.RAG_pipeline()

if definitions.cnvrg == False:
    definitions.external_language_model()


def prepare_prompt(prompt, documents, query):

    # replace documents
    prompt = prompt.replace("{documents}", documents)

    # replace query
    prompt = prompt.replace("{query}", query)

    return prompt


def preprocess(result, definitions):

    query = result["query"]
    documents = [
        result["documents"][i].content for i in range(0, len(result["documents"]))
    ]

    documents = " ".join(documents)

    return prepare_prompt(definitions.prompt_text, documents, query)

def query(data):

    params = {}
    query = data['query']

    result = definitions.pipeline.run(query=query, params=params, debug=False)

    preprocessed = preprocess(result, definitions)
    print(preprocessed)
    answer = definitions.call_llm(preprocessed)
    # return answer
    return answer
