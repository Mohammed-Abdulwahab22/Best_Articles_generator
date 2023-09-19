import os
from dotenv import find_dotenv, load_dotenv
import requests
import json

from langchain import PromptTemplate, LLMChain
import openai
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

# UI
import streamlit as st

load_dotenv(find_dotenv())
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")


# query = "summarise 2023 science articles"

# 1.serp request to get list of relevant articles
def search(query):
    url = "https://google.serper.dev/scholar"

    payload = json.dumps({"q": query})
    headers = {"X-API-KEY": SERPAPI_API_KEY,
               "Content-Type": "application/json"}
    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    print("search results: ", response_data)
    return response_data


# search(query)
# 2.llm to choose the best articles and return urls
def find_best_article_urls(response_data, query):
    # turn json into string
    response_str = json.dumps(response_data)

    # creat llm to choose the best article
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = f"""
    You are an articles researcher , you are extremely good at find most relevant articles to certain topic;
    {{response_str}}
    Above is the list of search results for the query {{query}}.
    Please choose the best 20 articles from the list, return ONLY an array of the urls, do not include anything else; return ONLY an array of the urls, do not include anything else
    
    """
    prompt_templates = PromptTemplate(input_variables=["response_str", "query"], template=template)
    article_picker_chain = LLMChain(llm=llm, prompt=prompt_templates, verbose=True)

    urls = article_picker_chain.predict(response_str=response_str, query=query)
    url_list = json.loads(urls)
    print(url_list)

    return url_list


# 3.get content for each article from urls and make summaries
def get_content_from_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    return data


def summarise(data, query):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=3000, chunk_overlap=200, length_function=len)
    text = text_splitter.split_documents(data)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = f"""
    {{text}}
    summarise the text above in order to create a summary about {{query}}
    Please follow all the following rules:
    1/ Make sure the summary is engaging, informative with good data
    2/ Make sure the summary is long and detailed, it should be no more than 20-30 paragraphs
    3/ The summary should address the {{query}} topic very well
    4/ The summary needs to be written in a way that is easy to read and understand
    5/ The summary needs to give audience actionable advice & insights too

    SUMMARY:
    """
    prompt_templates = PromptTemplate(input_variables=["text", "query"], template=template)

    summariser_chain = LLMChain(llm=llm, prompt=prompt_templates, verbose=True)

    summaries = []

    for chunk in enumerate(text):
        summary = summariser_chain.predict(text=chunk, query=query)
        summaries.append(summary)

    print(summaries)
    return summaries



def generate_thread(summaries, query):
    summaries_str = str(summaries)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.7)
    template = f""" 
    {{summaries_str}}
    
    text above is some context about {{query}} 
    Please write a long detailed summary about {{query}} using the text above, and following all rules below:
    1/ Make sure the summary is engaging, informative with good data
    2/ Make sure the summary is not too long, it should be no more than 20-30 paragraphs
    3/ The summary should address the {{query}} topic very well
    4/ The summary needs to be written in a way that is easy to read and understand
    5/ The summary needs to give audience actionable advice & insights too

    SUMMARY THREAD:
    """
    prompt_templates = PromptTemplate(input_variables=["summaries_str", "query"], template=template)
    sum_thread_chain = LLMChain(llm=llm, prompt=prompt_templates, verbose=True)

    sum_thread = sum_thread_chain.predict(summaries_str=summaries_str, query=query)

    return sum_thread


# build UI
def main():
    st.set_page_config(page_title="Fresh", page_icon="https://images-platform.99static.com//b958zSHpQuk9oliGKL3rWiV8s2k=/1x0:1500x1499/fit-in/500x500/projects-files/32/3210/321012/2ddaa18d-8221-4399-8043-cacfb356c657.jpg", layout="wide")
    st.image("https://images-platform.99static.com//b958zSHpQuk9oliGKL3rWiV8s2k=/1x0:1500x1499/fit-in/500x500/projects-files/32/3210/321012/2ddaa18d-8221-4399-8043-cacfb356c657.jpg", width=100)
    st.header("Generate the best articles: ")
    query = st.text_input("Topic of the article:")

    if query:
        print(query)
        st.write("Generating the best articles for: ", query)

        search_resluts = search(query + " articles")
        urls = find_best_article_urls(search_resluts, query)
        data = get_content_from_urls(urls)
        summaries = summarise(data, query)
        thread = generate_thread(summaries, query)



        with st.expander("The Best articles about the topic"):
            st.info(urls)


        with st.expander("Summary about the topic"):
            st.info(thread)


if __name__ == '__main__':
    main()
