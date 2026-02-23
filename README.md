# AI Singapore Secondary School Recommendation System

A RAG-based (Retrieval Augmented Generation) recommendation system that helps students find suitable secondary schools in Singapore based on their PSLE scores, location preferences, and CCA interests.

## Overview

This project uses LangChain, ChromaDB, and OpenAI to provide personalized school recommendations. It combines structured data filtering with semantic search, web search for travel information, and LLM-powered explanations. A Streamlit web app provides an interactive frontend.

## Features

- Filter schools by PSLE score and posting group eligibility
- Strict PSLE score-to-posting-group validation (4-20 → PG3, 21-22 → PG3/PG2, 23-24 → PG2, 25 → PG2/PG1, 26-30 → PG1)
- Weighted ranking based on location and CCA preferences
- Support for school type preferences (Co-ed, Boys', Girls')
- Zone and specific location filtering (North, South, East, West + 28 districts)
- CCA matching across 175 activities in 5 categories
- Web search via SerpAPI for nearest MRT stations and travel options
- Source references from both ChromaDB and web search in the output
- Natural language explanations powered by GPT-4o-mini
- Interactive Streamlit web interface
- RAGAS evaluation framework for quality assessment

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain (LCEL) |
| Vector Database | ChromaDB |
| Embeddings | OpenAI text-embedding-3-large |
| LLM | GPT-4o-mini |
| Web Search | SerpAPI (Google Search) |
| Frontend | Streamlit |
| Data Processing | Pandas |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision) |


## Data Sources

### schinfo.csv
School general information including:
- School name, address, postal code
- Contact details (phone, fax, email)
- Transportation (MRT, bus routes)
- Zone and district
- School type (Government, Independent, etc.)
- Special programs (SAP, Autonomous, Gifted, IP)
- Mother tongue offerings

### COP.csv
Cut-Off Points for school admission:
- IP (Integrated Programme) scores
- PG3 (Posting Group 3) scores
- PG2 (Posting Group 2) scores
- PG1 (Posting Group 1) scores
- Affiliated school indicator

### CCA.csv
Co-Curricular Activities offered:
- CCA grouping (e.g., Robotics, Basketball, Choir)
- CCA type (Physical Sports, Clubs and Societies, Visual and Performing Arts, Uniformed Groups)

### References
APP  - https://iti123projectai-school-recommendation-zlyhmesbj8yruxygk8debz.streamlit.app/

COP  - https://sgschooling.com/secondary/cop/all

CCAs - https://data.gov.sg/datasets?topics=education&query=sch+cca&resultId=d_9aba12b5527843afb0b2e8e4ed6ac6bd

SCH  - https://data.gov.sg/datasets?topics=education&query=sch+cca&resultId=d_688b934f82c1059ed0a6993d2a829089
