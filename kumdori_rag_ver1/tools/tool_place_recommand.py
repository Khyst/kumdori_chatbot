import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

import streamlit as st

from tavily import TavilyClient

class place_recommand: # 맛집, 관광지 등의 맛집 추천 툴
    
    def __init__(self):
        self.API_KEY = os.getenv("PLACES_API_KEY", "AIzaSyCUJvLApxRSiVGWou-_CHDOtiCc1yE_GYE")
    
    def search_restaurants(self, location_query):
        """
        Google Places API의 Text Search를 사용하여 맛집을 검색합니다.

        Args:
            location_query (str): 검색할 지역 및 키워드 (예: "판교동 맛집, 한국").

        Returns:
            list: 검색된 맛집 정보 리스트 또는 빈 리스트.
        """
        
        # Text Search API 엔드포인트
        url = 'https://places.googleapis.com/v1/places:searchText'
        
        # 요청 바디 (JSON 형태)
        data = {
          "textQuery" : location_query
        }
        
        # 헤더 설정 (API 키와 필드 마스크 포함)
        # 필요한 필드만 요청하여 비용을 절감합니다.
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.API_KEY,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.id,places.types,places.reviews'
        }
        
        print(f"INFO: 맛집 검색 요청. 쿼리: {location_query}")
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            
            result = response.json()
            
            # 검색 결과 (places 리스트)를 반환
            return result.get('places', [])
            
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Google Places API 요청 실패: {e}")
            return []

    def search_places(self, location_query):
        """
        Google Places API의 Text Search를 사용하여 관광지를 검색합니다.

        Args:
            location_query (str): 검색할 지역 및 키워드 (예: "판교동 관광지, 한국").

        Returns:
            list: 검색된 관광지 정보 리스트 또는 빈 리스트.
        """
        
        # Text Search API 엔드포인트
        url = 'https://places.googleapis.com/v1/places:searchText'
        
        # 요청 바디 (JSON 형태)
        data = {
          "textQuery" : location_query
        }
        
        # 헤더 설정 (API 키와 필드 마스크 포함)
        # 필요한 필드만 요청하여 비용을 절감합니다.
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.API_KEY,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.rating,places.priceLevel,places.id,places.types,places.reviews'
        }
        
        print(f"INFO: 관광지 검색 요청. 쿼리: {location_query}")
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            
            result = response.json()
            
            # 검색 결과 (places 리스트)를 반환
            return result.get('places', [])
            
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Google Places API 요청 실패: {e}")
            return []

