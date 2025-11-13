# """ 기본 라이브러리 """
import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

import streamlit as st

from tavily import TavilyClient

class weather_forecast: # 일기 예보를 조회하는 툴
    
    def __init__(self):
        # 광역시/도, 시/군/구, 동/읍/면, 날짜, 시간 정보를 바탕으로 날씨 예보를 조회합니다.
        self.xy_list = None  # 격자 좌표 데이터프레임
        
        self.load_grid_data()
        self.WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
    
    def load_grid_data(self):
        """
        제공된 XLSX 파일을 Pandas DataFrame으로 로드합니다.
        데이터 로드는 애플리케이션 실행 시 한 번만 수행되어야 합니다.
        
        데이터프레임의 컬럼 이름:
        '1단계' (시/도), '2단계' (시/군/구), '3단계' (동/읍/면), 
        '격자 X', '격자 Y', '경도(초/100)', '위도(초/100)'
        """
        
        filepath = os.path.join(os.path.dirname(__file__), "../xylist.xlsx")
        
        try:
            # read_excel 대신 read_csv를 사용해야 할 경우 read_csv로 변경하세요.
            df = pd.read_excel(filepath)
            
            # 컬럼 이름이 한글이므로 사용의 편의를 위해 영어로 변환합니다.
            df.rename(columns={
                '1단계': 'province',
                '2단계': 'city', 
                '3단계': 'region', 
                '격자 X': 'nx', 
                '격자 Y': 'ny', 
                '경도(초/100)': 'lon',
                '위도(초/100)': 'lat'
            }, inplace=True)
            
            # 격자 좌표와 위도/경도 컬럼이 숫자인지 확인
            df['nx'] = pd.to_numeric(df['nx'], errors='coerce')
            df['ny'] = pd.to_numeric(df['ny'], errors='coerce')
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            
            # NaN 값이 있는 행 제거 및 문자열 컬럼 정리
            self.xy_list = df.dropna(subset=['nx', 'ny', 'lat', 'lon']).copy()
            self.xy_list['province'] = self.xy_list['province'].fillna('').astype(str).str.strip()
            self.xy_list['city'] = self.xy_list['city'].fillna('').astype(str).str.strip()
            self.xy_list['region'] = self.xy_list['region'].fillna('').astype(str).str.strip()
            
            print("INFO: 날씨 격자 데이터 로드 완료.")
            print(f"INFO: 총 {len(self.xy_list)}개의 위치 데이터 로드됨.")
            return True

        except FileNotFoundError:
            print(f"ERROR: 격자 데이터 파일({filepath})을 찾을 수 없습니다. 경로를 확인해주세요.")
            return False
        
        except Exception as e:
            print(f"ERROR: 격자 데이터 로드 중 오류 발생: {e}")
            return False
    
    def normalize_city_name(self, province, city):
        """
        행정구역 통합/개편으로 인해 변경된 시/군/구 이름을 정규화합니다.
        """
        # 경상남도 통합창원시 관련 매핑
        if province == "경상남도":
            city_mappings = {
                "진해시": ["창원시진해구"],
                "마산시": ["창원시마산합포구", "창원시마산회원구"],
                "창원시": ["창원시의창구", "창원시성산구"]
            }
            
            if city in city_mappings:
                return city_mappings[city]
        
        # 다른 지역의 매핑이 필요하면 여기에 추가
        # 예: 전라남도, 충청북도 등의 통합 사례
        
        # 매핑되지 않은 경우 원본 반환
        return [city]
    
    def set_location(self, province, city, region):
        
        self.province = province
        self.city = city
        self.region = region
        
    def get_coordinates(self):
        """
        주어진 행정구역에 해당하는 격자 좌표(nx, ny)와 위도/경도(lat, lon)를 조회합니다.
        """
        
        # 데이터가 로드되지 않았다면 재시도 (운영 환경에서는 이 부분 제거 가능)
        if self.xy_list is None:
            if not self.load_grid_data():
                return None
        
        # None 값들을 문자열로 변환 및 공백 제거
        province = str(self.province).strip() if self.province and self.province != 'None' else ''
        city = str(self.city).strip() if self.city and self.city != 'None' else ''
        region = str(self.region).strip() if self.region and self.region != 'None' else ''
        
        # 도시 이름 정규화 (통합된 도시명으로 변환)
        possible_cities = self.normalize_city_name(province, city)
        
        # 각 가능한 도시명에 대해 좌표 검색 시도
        for normalized_city in possible_cities:
            # 지역명 필터링 (동/읍/면 단위로 검색하는 것이 가장 정확)
            # region이 비어있거나 'None'이 아닌 경우에만 region으로 필터링
            if region and region != 'None':
                query = self.xy_list[
                    (self.xy_list['province'] == province) &
                    (self.xy_list['city'] == normalized_city) &
                    (self.xy_list['region'] == region)
                ]
                
                if not query.empty:
                    # 첫 번째 일치하는 행의 데이터를 사용합니다.
                    row = query.iloc[0]
                    if normalized_city != city:
                        print(f"INFO: '{city}'는 '{normalized_city}'로 변경되었습니다. 변경된 지역의 좌표를 사용합니다.")
                    return {
                        'nx': row['nx'],
                        'ny': row['ny'],
                        'lat': row['lat'],
                        'lon': row['lon']
                    }
            
            # 동/읍/면 단위에서 못 찾았거나 region이 None인 경우 시/군/구 단위로 검색
            query = self.xy_list[
                (self.xy_list['province'] == province) &
                (self.xy_list['city'] == normalized_city)
            ]
            
            if not query.empty:
                # 시/군/구의 대표 지점 (예: 첫 번째 행)의 좌표를 사용합니다.
                row = query.iloc[0]
                if normalized_city != city:
                    print(f"INFO: '{city}'는 '{normalized_city}'로 변경되었습니다. 변경된 지역의 좌표를 사용합니다.")
                if region and region != 'None':
                    print(f"WARNING: '{region}'에 대한 정확한 좌표를 찾을 수 없어, '{normalized_city}'의 대표 좌표를 사용합니다.")
                else:
                    print(f"INFO: 동/구 정보가 없어 '{normalized_city}'의 대표 좌표를 사용합니다.")
                return {
                    'nx': row['nx'],
                    'ny': row['ny'],
                    'lat': row['lat'],
                    'lon': row['lon']
                }
        
        # 정규화된 도시명으로도 못 찾은 경우, 도/시 단위로 검색 (최후의 수단)
        query = self.xy_list[
            (self.xy_list['province'] == province)
        ]
        
        if not query.empty:
            row = query.iloc[0]
            print(f"WARNING: '{city}'에 대한 정확한 좌표를 찾을 수 없어, '{province}'의 대표 좌표를 사용합니다.")
            return {
                'nx': row['nx'],
                'ny': row['ny'],
                'lat': row['lat'],
                'lon': row['lon']
            }
            
        print(f"ERROR: '{province} {city} {region}'에 해당하는 좌표를 찾을 수 없습니다.")
        
        return None

    def get_current_datetime(self):
        """
        현재 날짜와 시간을 'yyyyMMdd' 및 'HHMM' 형식으로 반환
        기상청 API의 발표시간에 맞춰 조정
        
        Returns:
            tuple: (date_str, time_str)
        """
        # 한국 표준시(KST, UTC+9)로 현재 시각을 얻음
        now = datetime.now(timezone(timedelta(hours=9)))
        
        # 기상청 초단기예보 발표시간: 매시 30분에 발표 (1시간 후부터 6시간까지)
        # API 호출가능 시간: 발표시간 + 10분 후 (매시 40분 이후)
        
        # 현재 시간이 40분 이전이면 이전 시간 기준으로 설정
        if now.minute < 40:
            base_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        else:
            base_time = now.replace(minute=0, second=0, microsecond=0)
        
        # 혹시 모를 안전장치: 30분 전 시간 사용
        base_time = base_time - timedelta(minutes=30)
        
        date_str = base_time.strftime("%Y%m%d")
        time_str = base_time.strftime("%H00")
        
        print(f"DEBUG: 현재시각={now.strftime('%Y-%m-%d %H:%M')}, 요청기준시각={base_time.strftime('%Y-%m-%d %H:%M')}")
        
        return date_str, time_str
    
    def _retry_with_different_time(self, province, city, region, orig_date, orig_time, nx, ny, lat, lon):
        """
        NO_DATA 오류 시 다른 발표시간으로 재시도
        """
        print("INFO: 다른 발표시간으로 재시도 중...")
        
        # 현재 시간 기준으로 이전 몇 시간 시도
        now = datetime.now(timezone(timedelta(hours=9)))
        
        retry_times = []
        for hours_back in [1, 2, 3, 6]:
            retry_time = now - timedelta(hours=hours_back)
            retry_date = retry_time.strftime("%Y%m%d")
            retry_hour = retry_time.strftime("%H00")
            retry_times.append((retry_date, retry_hour))
        
        url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'
        
        for retry_date, retry_time in retry_times:
            print(f"INFO: 재시도 - base_date={retry_date}, base_time={retry_time}")
            
            params = {
                'serviceKey': os.getenv("WEATHER_API_KEY"),
                'pageNo': '1', 
                'numOfRows': '100', 
                'dataType': 'JSON', 
                'base_date': retry_date, 
                'base_time': retry_time, 
                'nx': str(int(nx)),
                'ny': str(int(ny))
            }
            
            try:
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if (data.get("response", {}).get("header", {}).get("resultCode") == "00" and
                        data.get("response", {}).get("body", {}).get("items", {}).get("item")):
                        
                        print(f"SUCCESS: {retry_date} {retry_time} 데이터로 성공!")
                        items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                        
                        # 날씨 데이터 처리 (기존 로직과 동일)
                        weather_info = {}
                        for item in items:
                            category = item.get("category")
                            fcstValue = item.get("fcstValue")
                            fcstTime = item.get("fcstTime")
                            
                            if fcstTime not in weather_info:
                                weather_info[fcstTime] = {}
                            
                            weather_info[fcstTime][category] = fcstValue
                        
                        weather_text = f"{province} {city} {region}의 {retry_date} {retry_time} 기준 날씨 예보\n\n"
                        
                        for fcstTime in sorted(weather_info.keys()):
                            info = weather_info[fcstTime]
                            weather_text += f"예보 시간: {fcstTime}시\n"
                            weather_text += "------------------------------------------------------------------------\n"
                            weather_text += f"- 기온(T1H): {info.get('T1H', 'N/A')} °C\n"
                            weather_text += f"- 강수확률(POP): {info.get('POP', 'N/A')} %\n"
                            weather_text += f"- 습도(REH): {info.get('REH', 'N/A')} %\n"
                            weather_text += f"- 풍속(WS10): {info.get('WS10', info.get('WDSD', 'N/A'))} m/s\n"
                            weather_text += f"- 하늘상태(SKY): {info.get('SKY', 'N/A')} (1: 맑음, 3: 구름많음, 4: 흐림)\n"
                            weather_text += "------------------------------------------------------------------------\n\n"
                        
                        st.write(weather_text)
                        return weather_text
                        
            except Exception as e:
                print(f"재시도 실패 ({retry_date} {retry_time}): {e}")
                continue
        
        # 모든 재시도 실패
        return f"죄송해요, 현재 {province} {city} {region} 지역의 날씨 정보를 가져올 수 없어요. 잠시 후 다시 시도해주세요."

    def get_weather_forcast(self, province, city, region):  
        
        self.set_location(province, city, region)
        
        coords = self.get_coordinates()
        
        date_str, time_str = self.get_current_datetime()
        
        if coords is None:
            error_msg = f"날씨 조회 실패: '{province} {city} {region}'에 해당하는 지역을 찾을 수 없습니다. 지역명을 다시 확인해주세요."
            st.error(error_msg)
            print(f"ERROR: {error_msg}")
            return error_msg

        nx = coords['nx']
        ny = coords['ny']
        lat = coords['lat']
        lon = coords['lon']
        
        print(f"조회 좌표: 격자 ({nx}, {ny}), 위도/경도 ({lat:.4f}, {lon:.4f})")
        
        # 기상청 단기 예보 API는 격자 좌표(nx, ny)를 사용하며, base_time은 발표 시간을 의미합니다. (기상청 단기 예보 API 호출 (Grid X, Grid Y 사용))
        url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst'
        
        params = {
            'serviceKey': os.getenv("WEATHER_API_KEY"),  # 기상청 API 키 (디코딩된 키 사용)
            'pageNo': '1', 
            'numOfRows': '100', 
            'dataType': 'JSON', 
            'base_date': date_str, 
            'base_time': time_str, 
            'nx': str(int(nx)),
            'ny': str(int(ny))
        }
        
        print("API Key: ", self.WEATHER_API_KEY)
        print(f"API 요청 URL: {url} / 기상청 동네 예보 API")
        print(f"API 파라미터: base_date={date_str}, base_time={time_str}, nx={int(nx)}, ny={int(ny)}, lat={lat:.4f}, lon={lon:.4f}")
        
        try:
            # API 호출 및 응답 처리
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"API 호출 실패. HTTP 상태 코드: {response.status_code}")
                print(f"응답 내용: {response.text[:500]}")
                return
            
            # JSON 파싱 시도
            try:
                data = response.json()
                
            except json.JSONDecodeError as json_err:
                print(f"JSON 파싱 실패: {json_err}")
                print("응답이 JSON 형식이 아닙니다. 응답 내용:")
                print(response.text[:1000])
                return
            
            if data.get("response", {}).get("header", {}).get("resultCode") != "00":
                error_code = data.get("response", {}).get("header", {}).get("resultCode")
                error_msg = data.get("response", {}).get("header", {}).get("resultMsg")
                print(f"API 오류: 코드={error_code}, 메시지={error_msg}")
                
                # NO_DATA 오류인 경우 다른 시간으로 재시도
                if error_code == "03" or "NO_DATA" in str(error_msg):
                    print("INFO: NO_DATA 오류 - 다른 발표시간으로 재시도합니다.")
                    return self._retry_with_different_time(province, city, region, date_str, time_str, nx, ny, lat, lon)
                
                return f"날씨 정보 조회 실패: {error_msg}"
            
            items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            
            if not items:
                print("INFO: 예보 데이터가 없습니다 - 다른 시간으로 재시도합니다.")
                return self._retry_with_different_time(province, city, region, date_str, time_str, nx, ny, lat, lon)
            
            # 필요한 정보 추출 및 출력
            weather_info = {}
            
            for item in items:
                category = item.get("category")
                fcstValue = item.get("fcstValue")
                fcstTime = item.get("fcstTime")
                
                if fcstTime not in weather_info:
                    weather_info[fcstTime] = {}
                
                weather_info[fcstTime][category] = fcstValue
                
            # 예보 시간별로 정렬하여 텍스트로 저장
            weather_text = f"{province} {city} {region}의 {date_str} {time_str} 기준 날씨 예보\n\n"
            
            for fcstTime in sorted(weather_info.keys()):
                
                info = weather_info[fcstTime]
                weather_text += f"예보 시간: {fcstTime}시\n"
                weather_text += "------------------------------------------------------------------------\n"
                weather_text += f"- 기온(T1H): {info.get('T1H', 'N/A')} °C\n"
                weather_text += f"- 강수확률(POP): {info.get('POP', 'N/A')} %\n"
                weather_text += f"- 습도(REH): {info.get('REH', 'N/A')} %\n"
                weather_text += f"- 풍속(WDSD): {info.get('WDSD', 'N/A')} m/s\n"
                weather_text += f"- 하늘상태(SKY): {info.get('SKY', 'N/A')} (1: 맑음, 3: 구름많음, 4: 흐림)\n"
                weather_text += "------------------------------------------------------------------------\n\n"
            
            print(weather_text)
            
            return weather_text

        except requests.exceptions.RequestException as e:
            print(f"네트워크 오류: {e}")
            return f"죄송해요, 네트워크 문제로 날씨 정보를 가져올 수 없어요. 인터넷 연결을 확인하고 다시 시도해주세요."
            
        except Exception as e:
            print(f"날씨 데이터 처리 오류: {e}")
            return f"죄송해요, 날씨 데이터를 처리하는 중 문제가 발생했어요. 잠시 후 다시 시도해주세요."

