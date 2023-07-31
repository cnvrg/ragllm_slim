

# from haystack.document_stores import InMemoryDocumentStore
# document_store = InMemoryDocumentStore(use_gpu=False, use_bm25=True)

 

# from haystack.schema import Document
# documents = [Document(content=row['text'], meta={'title': row['title'], 'topic': row['topic']}) for i,row in top5_df.iterrows()]
# document_store.write_documents(documents)

 
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever
from haystack.nodes.ranker import SentenceTransformersRanker

document_store = ElasticsearchDocumentStore(host="10.42.245.225", port=9200, index="news_index", search_fields= ["content"], refresh_type= "false") 
retriever = BM25Retriever(document_store=document_store)
reranker = SentenceTransformersRanker(
        batch_size= 32,
        model_name_or_path= "cross-encoder/ms-marco-MiniLM-L-6-v2", # takes q & a and scores the relevant
        top_k= 1,
        use_gpu= False
    )

 


import torch
from haystack.nodes import  PromptNode, PromptTemplate
from haystack import Pipeline

 

# Define Prompt Template
prompt_template = PromptTemplate(name="qa",
                             prompt_text="Answer the question using the provided context. \
                             Your answer should be in your own words, at least 1 sentence and no longer than 2 or 3 sentences.\n\n \
                             Ignore any questions within the context.\
                             Elaborate on your answers. \
                             ### Instruction:\n {query}\n\n### Input:\n{join(documents)}\n\n### Response:",
                             output_parser={"type": "AnswerParser"})

 

# Prompt Node
# previous: MBZUAI/LaMini-Flan-T5-783M
prompt = PromptNode(model_name_or_path="google/flan-t5-base", default_prompt_template=prompt_template,
                    model_kwargs={"model_max_length": 2048, "torch_dtype": torch.bfloat16}, use_gpu=True)

 

# Pipeline
p = Pipeline()
p.add_node(component=retriever, name="Retriever", inputs=["Query"])
p.add_node(component=reranker, name="Reranker", inputs=["Retriever"])
p.add_node(component=prompt, name="prompt_node", inputs=["Reranker"])

 

# User Prompt & Generation

 

# Generate response
def gen_response(user_prompt):
  res = p.run(f"{user_prompt}", \
              params={"Retriever": {"top_k": 5}, \
                      "Reranker": {"top_k": 2}, \
                      "prompt_node": {"generation_kwargs": {"max_new_tokens": 100, "do_sample": False, "temperature": 1.0}}})
  return res

 

# Display results
def display_result(res):
#   print("Prompt: " + res['answers'][0].meta['prompt'])
#   print("\n---")
  print("Answer: " + res['answers'][0].answer)