import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import ollama
print("Starting the sesion")
client = chromadb.PersistentClient(path="./RAG Tuturial/chroma_db")
embed_model = SentenceTransformer('BAAI/bge-small-en-v1.5', local_files_only=True)
#embed_model = SentenceTransformer('all-MiniLM-L6-v2')
collection = client.get_or_create_collection("policies")
def embedUserQuery(user_query):
    query_vec = embed_model.encode([user_query]).tolist()
    result = collection.query(query_embeddings=query_vec,n_results=10)
    return result
load_dotenv()

print(f"Starting the Session.......{datetime.now()}")

class HR_Assistant:
    def __init__(self,user_prompt, HR_Policies):
        self.user_prompt = user_prompt
        self.HR_Policies = HR_Policies
        self.My_key = os.getenv("GeminiAPIKey")
        #self.client = genai.Client(api_key=self.My_key)
    def askAi(self):
        try:
            context_text = "\n".join(self.HR_Policies)
            full_prompt = f""" You are a literal document-reading robot..
            Answer strictly using the context below.
            If the answer is not available in the context, say:
            "Please reach out to your respective HR."
            Do not use your own knowledge. Stick to the facts in the text.
            you should maintain conversation like a real HR ,let's say greeting and other human behaviours.
            """
            #Context: {context_text}
            #Question:{self.user_prompt}
            
            '''
            response = self.client.models.generate_content(
                #model="gemini-2.5-flash-lite",
                model="gemini-2.5-flash",
                contents=full_prompt,
                config= types.GenerateContentConfig(
                system_instruction="Do not use Markdown formatting (no asterisks, bolding, or headers). " \
                "Provide plain text only."
                )
            )
            '''
            response = ollama.chat(model='qwen2:0.5b', 
                messages=[
                    {
                        'role': 'system',
                        'content': full_prompt
                    },
                    {
                        'role': 'user',
                        'content': f"Context: {context_text}\n\nQuestion: {self.user_prompt}"
                    }
                ],
                options = {
                    'temperature':0.0,
                }
            ) 
            
            print(f"Ollama Ai Bot: {response['message']['content']}")
            
            #print(f"Ai Bot: {response.text}")
            print(f"ending the Process.......{datetime.now()}")
        except Exception as e:
            print(f"An error occurred: {e}")




while (True):
    print("Enter your question below. If you want to stop enter 'quit' or 'exit' or click enter.")
    user_query = input("Enter you question: ")
    if(user_query.lower() in ['exit','quit'] or user_query == ""):
        print("Stopping the session.")
        break
    result = embedUserQuery(user_query)
    
    prompt1 = HR_Assistant(user_query, result["documents"][0])
    prompt1.askAi()


