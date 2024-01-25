from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os

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

class LoadUrlsAPI(APIView):

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
    




class LoadTrainData(APIView):
         
        def get(self, request): 
            csv_path = os.path.join(os.path.dirname(__file__), '../../leetcode_dataset.csv')
            loader = TextLoader(csv_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(docs, embeddings)
            db.save_local("problemset")
            return Response({ "message": 'done'})
        
from langchain_community.vectorstores import AstraDB     

ASTRA_DB_API_ENDPOINT =  os.environ.get('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN = os.environ.get('ASTRA_DB_APPLICATION_TOKEN')
class LoadTrainDataWithAstraDB(APIView):
         
        def get(self, request): 

            
            csv_path = os.path.join(os.path.dirname(__file__), '../../leetcode_dataset.csv')
            loader = TextLoader(csv_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            embeddings = OpenAIEmbeddings()

            vstore = AstraDB(
                embedding=embeddings,
                collection_name="problemset",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )
            inserted_ids = vstore.add_documents(docs)
            print(inserted_ids)
            return Response({ "message": 'done', "data": inserted_ids })
        


#  template = """
#                 **Title/Topic:**
#                 Algorithmic Problem

#                 **Description:**
#                 You are an experienced problem setter with a broad understanding of various programming concepts. 
#                 Your task is to generate a new coding problem for an interview.
#                 Keep the problem diverse, covering different areas of algorithms and data structures.

#                   Give the output as json.

#                 **CONTEXT**
#                 {context}
                
#                 **Question Level:**
#                 {level}

#                 **Desired Output/Example:**
#                 Your answers must be concise, in JSON and in the given format as follows: 

#                 - problem_description
#                 - example_inputs
#                 - example_outputs
#                 - input_output_explanation
#                 - placeholder_code
#                 - solutions
#             """

from datetime import datetime
class SuggestCodingProblem(APIView):
     
     def get(self, request):
          
            level = request.GET.get('level')
            goal = request.GET.get('goal')

            llm = ChatOpenAI(temperature=0.4, model_name='gpt-3.5-turbo')
            # db = FAISS.load_local("problemset", OpenAIEmbeddings())

            # retriever = db.as_retriever()

            embeddings = OpenAIEmbeddings()

            vstore = AstraDB(
                embedding=embeddings,
                collection_name="problemset",
                api_endpoint=ASTRA_DB_API_ENDPOINT,
                token=ASTRA_DB_APPLICATION_TOKEN,
            )

            retriever = vstore.as_retriever(search_kwargs={"k": 3})
            

            template = """
                **Generate a Coding Problem Set for Growth**

                **Description:**
                You are an aspiring developer looking to enhance your coding skills over the next {growth_duration}. Your current experience level is {experience}.

                **Your Task:**
                Generate a guided learning plan that includes topics, recommended duration for each topic and an overall learning duration goal ({growth_duration}). 
                For each learning topic, provide a brief introduction, recommend a duration, and include recommended number of coding problems for the user to solve. 
                The goal is to guide the user through learning the topics and applying the knowledge to solve problems within the specified overall duration.
                The topic set must be relevant to the given {goal}. While generating problem sets, only take problems from the given context. 
                Keep the problem sets diversified and atleast 5 problems per topic. Generate problems that are unique, not something that you can
                search and find at any search engines. For creating unique problems analyse the given context properly.

                **GOAL**
                {goal}

                **CONTEXT**
                {context}

                **Desired Output/Example:**
                Provide a structured learning plan with topics, recommended duration, and corresponding coding problems in JSON Format. 
                Each topic section should include:

                - topic
                - introduction
                - recommended_duration
                - Coding Problems: -> array of problems
                    - Problem summarized description (under 100 words)
                - ...

                The learning plan should guide the user through understanding and practicing each topic within the overall goal of {growth_duration}.

            """

            prompt = ChatPromptTemplate.from_template(template)
            # answer = db.similarity_search(question)
            chain = (
                prompt
                | llm
                | JsonOutputParser()
            )
            print(retriever)
           
            answer = chain.invoke({
                  "context": retriever,
                  "growth_duration": level,
                  "experience": "0.5 years",
                  "goal": goal,
            })
            return Response({ "message": answer })
    



    # node            - 300
    # laravel         - 300
    # asp net         - 300 
    # python/django   - 300
    # nextjs          - 300
    # nestjs         
    # rubyonralis     - 200
    # problem-solving - 2000
        # intro algorithm
            # time
            # space complexity
            # patterns
                # sliding_window
                # two pointers
                #  
        # adhoc problems
            # 
        # greedy
        # dynamic problems
        # graph
        # datastructures
            # stack
            # queue
            # linkedlist
            # queue
            # pointer
        # sorting
        # searching
        # backtracking
        # trees
        # tries
     

     # progress -> ranking -> top 10
        # talent base -> job market apply

    # system-design   - 100


    # How much time do you want to spend per day
    # Daily - monthly basis - packacge (15 days )

    # quiz - queston, 4 - option, correct answer (1/multiple)
    # problem solving - title, description, sample input, sample output, example code, boilerplate code
    # system design - problem description, answer
    # technical questions - problem description, boilerplate code