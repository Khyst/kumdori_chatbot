import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

import streamlit as st

from tavily import TavilyClient

# """ Helper Classes """
class korea_regions_verificator:
    """
    한국 법정동 코드를 기반으로 정확한 지역명을 검증하고 추천하는 헬퍼 클래스
    """
    
    def __init__(self):
        self.regions_df = None
        
        self.load_regions_data()
    
    def load_regions_data(self):
        """korea_regions.csv 파일을 로드합니다."""
        try:
            filepath = os.path.join(os.path.dirname(__file__), "../korea_regions.csv")
            self.regions_df = pd.read_csv(filepath)
            
            # 빈 값들을 빈 문자열로 처리
            self.regions_df = self.regions_df.fillna('')
            
            print("INFO: 한국 법정구역 데이터 로드 완료.")
            print(f"INFO: 총 {len(self.regions_df)}개의 법정구역 데이터 로드됨.")
            return True
            
        except Exception as e:
            print(f"ERROR: 한국 법정구역 데이터 로드 실패: {e}")
            return False
    
    def get_valid_provinces(self):
        """유효한 시도명 목록을 반환합니다."""
        if self.regions_df is None:
            return []
        
        # 현재 사용되는 시도명만 추출 (과거 명칭 제외)
        current_provinces = [
            "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", 
            "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", 
            "충청북도", "충청남도", "전북특별자치도", "전라남도", "경상북도", 
            "경상남도", "제주특별자치도"
        ]
        
        return [p for p in current_provinces if p in self.regions_df['시도명'].values]
    
    def get_all_valid_cities(self):
        """모든 유효한 시군구명 목록을 반환합니다."""
        if self.regions_df is None:
            return []
        
        cities = self.regions_df[self.regions_df['시군구명'] != '']['시군구명'].unique().tolist()
        
        return sorted(cities)
    
    def get_all_valid_regions(self):
        """모든 유효한 읍면동명 목록을 반환합니다."""
        if self.regions_df is None:
            return []
        
        regions = self.regions_df[self.regions_df['읍면동명'] != '']['읍면동명'].unique().tolist()
        
        return sorted(regions)
    
    def get_all_cities(self):
        """모든 유효한 시군구명 목록을 반환합니다 (별칭)."""
        return self.get_all_valid_cities()
    
    def get_all_regions(self):
        """모든 유효한 읍면동명 목록을 반환합니다 (별칭)."""
        return self.get_all_valid_regions()
    
    def get_all_regions_for_city(self, city):
        """특정 시군구에 속하는 모든 읍면동명 목록을 반환합니다."""
        if self.regions_df is None or not city:
            return []
        
        regions = self.regions_df[
            (self.regions_df['시군구명'] == city) & 
            (self.regions_df['읍면동명'] != '')
        ]['읍면동명'].unique().tolist()
        
        return sorted(regions)
    
    def get_province_for_city(self, city):
        """특정 시군구가 속한 시도를 반환합니다."""
        if self.regions_df is None or not city:
            return None
        
        matches = self.regions_df[self.regions_df['시군구명'] == city]
        
        if not matches.empty:
            return matches.iloc[0]['시도명']
        
        return None
    
    def get_location_for_region(self, region):
        """특정 읍면동이 속한 시도와 시군구를 반환합니다."""
        if self.regions_df is None or not region:
            return None
        
        matches = self.regions_df[self.regions_df['읍면동명'] == region]
        
        if not matches.empty:
            # 여러 개가 있을 수 있으므로 첫 번째 것을 반환
            row = matches.iloc[0]
            return {
                'province': row['시도명'],
                'city': row['시군구명']
            }
        
        return None
    
    def get_valid_cities_for_province(self, province):
        """특정 시도에 속하는 유효한 시군구명 목록을 반환합니다."""
        if self.regions_df is None or not province:
            return []
        
        cities = self.regions_df[
            (self.regions_df['시도명'] == province) & 
            (self.regions_df['시군구명'] != '')
        ]['시군구명'].unique().tolist()
        
        print(f"DEBUG: {province}의 유효한 시군구명 목록: {cities}")
        
        return sorted(cities)
    
    def get_valid_regions_for_city(self, province, city):
        """특정 시도, 시군구에 속하는 유효한 읍면동명 목록을 반환합니다."""
        if self.regions_df is None or not province or not city:
            return []
        
        regions = self.regions_df[
            (self.regions_df['시도명'] == province) & 
            (self.regions_df['시군구명'] == city) & 
            (self.regions_df['읍면동명'] != '')
        ]['읍면동명'].unique().tolist()
        
        print(f"DEBUG: {province} {city}의 유효한 읍면동명 목록: {regions}")
        
        return sorted(regions)
    
    def validate_location(self, province=None, city=None, region=None):
        
        """
        입력된 지역명이 유효한지 검증하고, 가능한 대안을 제시합니다.
        """
        if self.regions_df is None:
            return {"valid": False, "message": "지역 데이터를 로드할 수 없습니다."}
        
        result = {"valid": True, "corrections": {}, "suggestions": []}
        
        # 1. 시도 검증
        valid_provinces = self.get_valid_provinces()
        
        if province and province not in valid_provinces:
            
            result["valid"] = False
            
            result["corrections"]["province"] = f"'{province}'는 유효하지 않은 시도명입니다."
            
            # 유사한 시도명 찾기 (개선된 매핑)
            province_mappings = {
                "강원도": "강원특별자치도",
                "전라북도": "전북특별자치도", 
                "전북도": "전북특별자치도",
                "부산시": "부산광역시",
                "대구시": "대구광역시", 
                "인천시": "인천광역시",
                "광주시": "광주광역시",
                "대전시": "대전광역시", 
                "울산시": "울산광역시"
            }
            
            if province in province_mappings:
                result["suggestions"].append(f"'{province}' → '{province_mappings[province]}'를 의미하시나요?")
            else:
                # 부분 일치 검색
                for valid_province in valid_provinces:
                    if province in valid_province or valid_province in province:
                        result["suggestions"].append(f"'{province}' → '{valid_province}'를 의미하시나요?")
                        break
        
        # 2. 시군구 검증 (시도가 유효한 경우에만)
        if province and province in valid_provinces and city:
            valid_cities = self.get_valid_cities_for_province(province)
            if city not in valid_cities:
                result["valid"] = False
                result["corrections"]["city"] = f"'{city}'는 '{province}'에 없는 시군구명입니다."
                # 유사한 시군구명 찾기
                for valid_city in valid_cities:
                    if city in valid_city or valid_city in city or self._similar_names(city, valid_city):
                        result["suggestions"].append(f"'{city}' → '{valid_city}'를 의미하시나요?")
                        break
        
        # 3. 읍면동 검증 (시도, 시군구가 유효한 경우에만)
        if (province and province in valid_provinces and 
            city and city in self.get_valid_cities_for_province(province) and 
            region):
            valid_regions = self.get_valid_regions_for_city(province, city)
            if region not in valid_regions:
                result["valid"] = False
                result["corrections"]["region"] = f"'{region}'는 '{province} {city}'에 없는 읍면동명입니다."
                
                # 동명이 다른 지역에 있는지 확인
                other_locations = self._find_region_in_other_locations(region)
                if other_locations:
                    result["suggestions"].append(f"'{region}'는 다음 지역에 있습니다: {', '.join(other_locations)}")
                
                # 유사한 읍면동명 찾기
                for valid_region in valid_regions:
                    if region in valid_region or valid_region in region or self._similar_names(region, valid_region):
                        result["suggestions"].append(f"'{province} {city}'의 '{region}' → '{valid_region}'를 의미하시나요?")
                        break
        
        return result
    
    def _find_region_in_other_locations(self, region_name):
        """특정 동명이 다른 지역에 있는지 찾는 헬퍼 함수"""
        if self.regions_df is None or not region_name:
            return []
        
        matches = self.regions_df[self.regions_df['읍면동명'] == region_name]
        locations = []
        
        for _, row in matches.iterrows():
            location = f"{row['시도명']} {row['시군구명']}"
            if location not in locations:
                locations.append(location)
        
        return locations[:3]  # 최대 3개까지만 반환
    
    def _similar_names(self, name1, name2):
        """두 지역명이 유사한지 검사하는 헬퍼 함수"""
        if not name1 or not name2:
            return False
        
        # 길이 차이가 2 이상이면 유사하지 않다고 판단
        if abs(len(name1) - len(name2)) > 2:
            return False
        
        # 공통 문자가 50% 이상이면 유사하다고 판단
        common_chars = set(name1) & set(name2)
        similarity = len(common_chars) / max(len(set(name1)), len(set(name2)))
        
        return similarity >= 0.5
