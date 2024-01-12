from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

urls = [
    # "http://laravel.com/",
    # "http://laravel.com/docs/10.x/releases",
    # "http://laravel.com/docs/10.x/upgrade",
    # "http://laravel.com/docs/10.x/contributions",
    # "http://laravel.com/docs/10.x/installation",
    # "http://laravel.com/docs/10.x/configuration",
    "http://laravel.com/docs/10.x/structure",
    "http://laravel.com/docs/10.x/frontend",
    "http://laravel.com/docs/10.x/starter-kits",
    "http://laravel.com/docs/10.x/deployment",
    "http://laravel.com/docs/10.x/lifecycle",
    "http://laravel.com/docs/10.x/container",
    "http://laravel.com/docs/10.x/providers",
    "http://laravel.com/docs/10.x/facades",
    "http://laravel.com/docs/10.x/routing",
    "http://laravel.com/docs/10.x/middleware",
    "http://laravel.com/docs/10.x/csrf",
    "http://laravel.com/docs/10.x/controllers",
    "http://laravel.com/docs/10.x/requests",
    "http://laravel.com/docs/10.x/responses",
    "http://laravel.com/docs/10.x/views",
    "http://laravel.com/docs/10.x/blade",
    "http://laravel.com/docs/10.x/vite",
    "http://laravel.com/docs/10.x/urls",
    "http://laravel.com/docs/10.x/session",
    "http://laravel.com/docs/10.x/validation",
    "http://laravel.com/docs/10.x/errors",
    "http://laravel.com/docs/10.x/logging",
    "http://laravel.com/docs/10.x/artisan",
    "http://laravel.com/docs/10.x/broadcasting",
    "http://laravel.com/docs/10.x/cache",
    "http://laravel.com/docs/10.x/collections",
    "http://laravel.com/docs/10.x/contracts",
    "http://laravel.com/docs/10.x/events",
    "http://laravel.com/docs/10.x/filesystem",
    "http://laravel.com/docs/10.x/helpers",
    "http://laravel.com/docs/10.x/http-client",
    "http://laravel.com/docs/10.x/localization",
    "http://laravel.com/docs/10.x/mail",
    # "http://laravel.com/docs/10.x/notifications",
    # "http://laravel.com/docs/10.x/packages",
    # "http://laravel.com/docs/10.x/processes",
    # "http://laravel.com/docs/10.x/queues",
    # "http://laravel.com/docs/10.x/rate-limiting",
    # "http://laravel.com/docs/10.x/strings",
    # "http://laravel.com/docs/10.x/scheduling",
    # "http://laravel.com/docs/10.x/authentication",
    # "http://laravel.com/docs/10.x/authorization",
    # "http://laravel.com/docs/10.x/verification",
    # "http://laravel.com/docs/10.x/encryption",
    # "http://laravel.com/docs/10.x/hashing",
    # "http://laravel.com/docs/10.x/passwords",
    # "http://laravel.com/docs/10.x/database",
    # "http://laravel.com/docs/10.x/queries",
    # "http://laravel.com/docs/10.x/pagination",
    # "http://laravel.com/docs/10.x/migrations",
    # "http://laravel.com/docs/10.x/seeding",
    # "http://laravel.com/docs/10.x/redis",
    # "http://laravel.com/docs/10.x/eloquent",
    # "http://laravel.com/docs/10.x/eloquent-relationships",
    # "http://laravel.com/docs/10.x/eloquent-collections",
    # "http://laravel.com/docs/10.x/eloquent-mutators",
    # "http://laravel.com/docs/10.x/eloquent-resources",
    # "http://laravel.com/docs/10.x/eloquent-serialization",
    # "http://laravel.com/docs/10.x/eloquent-factories",
    # "http://laravel.com/docs/10.x/testing",
    # "http://laravel.com/docs/10.x/http-tests",
    # "http://laravel.com/docs/10.x/console-tests",
    # "http://laravel.com/docs/10.x/dusk",
    # "http://laravel.com/docs/10.x/database-testing",
    # "http://laravel.com/docs/10.x/mocking",
    # "http://laravel.com/docs/10.x/starter-kits#laravel-breeze",
    # "http://laravel.com/docs/10.x/billing",
    # "http://laravel.com/docs/10.x/cashier-paddle",
    # "http://laravel.com/docs/10.x/envoy",
    # "http://laravel.com/docs/10.x/fortify",
    # "http://laravel.com/docs/10.x/folio",
    # "http://laravel.com/docs/10.x/homestead",
    # "http://laravel.com/docs/10.x/horizon",
    # "http://laravel.com/docs/10.x/mix",
    # "http://laravel.com/docs/10.x/octane",
    # "http://laravel.com/docs/10.x/passport",
    # "http://laravel.com/docs/10.x/pennant",
    # "http://laravel.com/docs/10.x/pint",
    # "http://laravel.com/docs/10.x/precognition",
    # "http://laravel.com/docs/10.x/prompts",
    # "http://laravel.com/docs/10.x/pulse",
    # "http://laravel.com/docs/10.x/sail",
    # "http://laravel.com/docs/10.x/sanctum",
    # "http://laravel.com/docs/10.x/scout",
    # "http://laravel.com/docs/10.x/socialite",
    # "http://laravel.com/docs/10.x/telescope",
    # "http://laravel.com/docs/10.x/valet",
    # "http://laravel.com/api/10.x",
    # "http://laravel.com",
    # "http://laravel.com/docs/10.x/configuration#environment-configuration",
    # "http://laravel.com/docs/10.x/sail#using-devcontainers",
    # "http://laravel.com/docs/10.x/starter-kits#breeze-and-next",
    # "http://laravel.com/team",
    # "http://laravel.com/trademark",
]

class UrlLoaderApi(APIView):

    def get(self, request):
        # Load data from given websites as url
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split texts using separator and get documents
        text_splitter = CharacterTextSplitter(separator='\n',
                                              chunk_size=1000,
                                              chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        print(len(docs))
        # Store in faiss storage
        embeddings = OpenAIEmbeddings()
        vectorStore_openAI = FAISS.from_documents(docs, embeddings)
        vectorStore_openAI.save_local("faiss_store")

        return Response({ "message": "Data retrieved", "data": data }, status=status.HTTP_200_OK)
    

class AskAI(APIView):

    def get(self, request):
            question = request.GET.get('question')

            llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
            db = FAISS.load_local("faiss_store", OpenAIEmbeddings())

            retriever = db.as_retriever()
            # philo_template = """
            # You are a brilliant laravel developer that knows deep into laravel and able to 
            # to craft well-thought answers to user questions regarding core laravel. Use the provided context as 
            # the basis for your answers and do not make up new reasoning paths - just mix-and-match what you are given.
            # Your answers must be concise, in HTML format and to the point, and refrain from answering about other topics than laravel.
            # Also include code examples with the answers. 

            # CONTEXT:
            # {context}

            # QUESTION: {question}

            # YOUR ANSWER:"""

            template = """
                **Title/Topic:**
                Laravel

                **Description:**
                You are a brilliant laravel developer that knows deep into laravel and able to 
                to craft well-thought answers to user questions regarding core laravel. Use the provided context as 
                the basis for your answers and do not make up new reasoning paths - just mix-and-match what you are given.
                Your answers must be concise, in HTML format and to the point, and refrain from answering about other topics than laravel.
                Also include code examples with the answers. 

                **CONTEXT**
                {context}
                
                **Specific Question(s):**
                {question}

                **Desired Output/Example:**
                Your answers must be concise, in HTML format and to the point, and refrain from answering about other topics than laravel.
                Also include code examples with the answers. 
            """

            philo_prompt = ChatPromptTemplate.from_template(template)
            # answer = db.similarity_search(question)
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | philo_prompt
                | llm
                | StrOutputParser()
            )

            answer = chain.invoke(question)
            return Response({ "message": answer })
    



    # node            - 300
    # laravel         - 300
    # asp net         - 300 
    # python/django   - 300
    # nextjs          - 300
    # rubyonralis     - 200
    # problem-solving - 2000
    # system-design   - 100


    # How much time do you want to spend per day
    # Daily - monthly basis - packacge (15 days )

    # quiz - queston, 4 - option, correct answer (1/multiple)
    # problem solving - title, description, sample input, sample output, example code, boilerplate code
    # system design - problem description, answer
    # technical questions - problem description, boilerplate code