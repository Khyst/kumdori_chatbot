import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

import streamlit as st

from tavily import TavilyClient

# """ 각종 역할을 가지고 있는 LLM 보조 툴들 """
class web_search: # 웹 검색 하는 툴 
    
    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def search(self, query):
        

        search_response = self.client.search(
                        query=query,
                        search_depth="advanced",
                    )
        
        return search_response
