import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Searches users by hobbies and provides their full info in JSON format:
#   Input: `I need people who love to go to mountains`
#   Output:
#     ```json
#       "rock climbing": [{full user info JSON},...],
#       "hiking": [{full user info JSON},...],
#       "camping": [{full user info JSON},...]
#     ```
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in User Service will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor on
#    the retrieval step, we will remove deleted users and add new - it will also resolve the issue with consistency
#    within this 2 services and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.
# ---
# Useful links:
# Chroma DB: https://docs.langchain.com/oss/python/integrations/vectorstores/index#chroma
# Document#id: https://docs.langchain.com/oss/python/langchain/knowledge-base#1-documents-and-document-loaders
# Chroma DB, async add documents: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.aadd_documents
# Chroma DB, get all records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.get
# Chroma DB, delete records: https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.delete
# ---
# TASK:
# Implement such application as described on the `flow.png` with adaptive vector based grounding and 'lite' version of
# output grounding (verification that such user exist and fetch full user info)

class HobbiesResponse(BaseModel):
    hobbies: dict[str, list[int]] = Field(description="Dictionary mapping a hobby string to a list of user IDs who enjoy that hobby")

EXTRACTION_PROMPT = """You are an expert entity extraction system.

## Task:
Extract hobbies mentioned in the user's request and find users from the provided context who enjoy those hobbies.

## RAG CONTEXT:
{context}

## USER QUESTION:
{query}

## Instructions:
1. Identify the core hobbies or activities the user is asking about.
2. For each identified hobby, find the users in the context whose 'about_me' section indicates they enjoy it.
3. Extract only the 'id' of those matching users.
4. Output the result matching the provided JSON schema: a dictionary where keys are hobbies and values are lists of user IDs.
5. Do not include users who do not explicitly match the requested hobbies.

{format_instructions}
"""

class InOutGroundingApp:
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small-1",
            openai_api_version="2023-05-15",
            dimensions=384,
            azure_endpoint=DIAL_URL,
            api_key=API_KEY
        )
        self.llm_client = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="",
            temperature=0.0,
            azure_endpoint=DIAL_URL,
            api_key=API_KEY
        )
        self.user_client = UserClient()
        self.vectorstore = Chroma(
            collection_name="users_about_me",
            embedding_function=self.embeddings
        )
        self.parser = PydanticOutputParser(pydantic_object=HobbiesResponse)
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(EXTRACTION_PROMPT)
        ]).partial(format_instructions=self.parser.get_format_instructions())

    async def _sync_users(self):
        print("🔄 Syncing users with Chroma DB...")
        current_users = self.user_client.get_all_users()
        current_user_ids = {str(u['id']) for u in current_users}
        
        db_records = self.vectorstore.get()
        db_user_ids = set(db_records['ids']) if db_records and 'ids' in db_records else set()
        
        # Find deleted users
        deleted_ids = list(db_user_ids - current_user_ids)
        if deleted_ids:
            self.vectorstore.delete(ids=deleted_ids)
            print(f"Removed {len(deleted_ids)} deleted users from DB.")
            
        # Find new users
        new_ids = current_user_ids - db_user_ids
        if new_ids:
            new_docs = []
            for u in current_users:
                if str(u['id']) in new_ids:
                    # Embed only ID and about_me
                    content = f"ID: {u['id']}\nAbout Me: {u['about_me']}"
                    doc = Document(page_content=content, id=str(u['id']))
                    new_docs.append(doc)
            
            # Batch add documents
            batch_size = 100
            for i in range(0, len(new_docs), batch_size):
                await self.vectorstore.aadd_documents(new_docs[i:i+batch_size])
            print(f"Added {len(new_docs)} new users to DB.")
            
        print("✅ Sync complete.")
        return current_users

    async def process_query(self, query: str):
        # 1. Sync DB and get full user list
        all_users = await self._sync_users()
        user_dict = {u['id']: u for u in all_users}
        
        # 2. Retrieve relevant context (vector search)
        print("🔎 Retrieving relevant profiles...")
        results = await self.vectorstore.asimilarity_search(query, k=20)
        
        if not results:
            print("No relevant profiles found.")
            return {}
            
        context_str = "\n\n".join([doc.page_content for doc in results])
        
        # 3. LLM Extraction (Named Entity Extraction)
        print("🧠 Extracting entities via LLM...")
        chain = self.prompt_template | self.llm_client | self.parser
        extraction: HobbiesResponse = await chain.ainvoke({
            "context": context_str,
            "query": query
        })
        
        # 4. Output Grounding
        print("🔗 Performing output grounding...")
        final_result = {}
        for hobby, ids in extraction.hobbies.items():
            valid_users = []
            for uid in ids:
                if uid in user_dict:
                    valid_users.append(user_dict[uid])
                else:
                    # Fallback if DB sync race condition occurred or LLM hallucinated
                    try:
                        u = await self.user_client.get_user(uid)
                        valid_users.append(u)
                    except Exception:
                        print(f"Warning: User ID {uid} not found during output grounding.")
            
            if valid_users:
                final_result[hobby] = valid_users
                
        return final_result

async def main():
    app = InOutGroundingApp()
    
    print("\nWelcome to the Hobbies Searching Wizard!")
    print("Example: 'I need people who love to go to mountains and like painting'")
    
    while True:
        user_question = input("\n> ").strip()
        if user_question.lower() in ['quit', 'exit']:
            break
            
        result = await app.process_query(user_question)
        
        import json
        print("\nFinal Output:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
