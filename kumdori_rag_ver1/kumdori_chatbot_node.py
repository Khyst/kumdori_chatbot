# """ 기본 라이브러리 """
import os
import sys
import json
import requests

# """ Third-party 라이브러리 """
from enum import Enum
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# """ LangChain 관련 라이브러리 """
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import ChatMessage
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser, EnumOutputParser

# """ Langchain 관련 외부 Tools 라이브러리 """
from tavily import TavilyClient

# """ Streamlit GUI 라이브러리 """
import streamlit as st

# """ 전역 변수 및 상수 정의 """
PERSONA_PROMPT = """당신은 한국어에 능통한 친절한 챗봇입니다. 사용자가 질문하면 사용자의 질문에 대한 답변을 제공해야 합니다. 한국어로 아이에게 애기하듯이 말해주세요, 추후 목소리로 말할 수 있는 기능에 대비하여 자연스럽고 부드럽게 말해주세요. 없는 정보는 애기하지 말고, 모르면 모른다고 말하세요. 잘못된 정보를 제시하면 $100의 벌금을 부과할 겁니다, 검색한 정보에 대해서는 관련 링크를 같이 제시하면 좋아, 최종 답변은 사람에게 말하듯 하는 답변이어야 돼"""

CATEGORIZE_PROMPT = "입력한 문장을 분석하여, 다음의 카테고리 리스트에서 가장 가까운 카테고리 하나를 선택하시오.\n 카테고리 리스트: {categories}\n 출력 포맷:{format_instructions} \n\n 입력:{query}"
GET_PROVINCE_CITY_PROMPT = "입력한 문장을 분석하여, 한국의 시/도 단위 지역과 시/군/구 단위 지역 그리고 동/읍/면 단위 지역을 각각 하나씩 선택하시오. 둘 중 하나라도 추출할 수 없다면 None을 출력하시오. 실제로 존재하지 않는 지역명은 반드시 None이라고 출력해야 함 \n 출력 포맷:{format_instructions} \n\n 입력:{query}"

CATEGORIES = ["맛집", "관광지", "날씨", "검색", "현재 시간", "현재 날짜", "교통"]

st.title("💬")

# """ 각종 역할을 가지고 있는 LLM 체인들 """
def chatbot_llm_chain():
    prompt = PromptTemplate.from_template(
        template = PERSONA_PROMPT + "\n\n\n 관련 정보: {context} \n\n\n 사용자 요청: {user_input} \n 꿈돌이 로봇:"
    )
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | model
        
    return chain

def categorize_llm_chain():
    
    response_schemas = [
        ResponseSchema(name="category", description="정의된 카테고리들 중 선택된 하나의 카테고리", type="string")
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate.from_template(
        template = CATEGORIZE_PROMPT,
        partial_variables={"format_instructions": format_instructions, "categories": CATEGORIES},
    )
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | model | output_parser
    
    return chain

def region_llm_chain():
    
    # 한국 지역 데이터 헬퍼 인스턴스 생성
    regions_helper = korea_regions_helper()
    
    # 유효한 지역명 목록 가져오기
    valid_provinces = regions_helper.get_valid_provinces()
    
    # 프롬프트에 유효한 지역명 정보 포함
    enhanced_prompt = f"""
입력한 문장을 분석하여, 한국의 시/도 단위 지역과 시/군/구 단위 지역 그리고 동/읍/면 단위 지역을 각각 하나씩 선택하시오. 
둘 중 하나라도 추출할 수 없다면 None을 출력하시오. 
실제로 존재하지 않는 지역명은 반드시 None이라고 출력해야 함.

현재 유효한 시/도명 목록:
{', '.join(valid_provinces)}

중요한 지역명 매핑 규칙:
1. 문지동, 탑립동 → 대전광역시 유성구 (서울이 아님!)
2. 판교동 → 경기도 성남시 분당구
3. 역삼동, 삼성동, 청담동 → 서울특별시 강남구
4. 강남역 주변 → 서울특별시 강남구 역삼동
5. 홍대 → 서울특별시 마포구 서교동
6. 명동 → 서울특별시 중구 명동
7. 신촌 → 서울특별시 서대문구 창천동

주의사항:
1. 동명이 같더라도 반드시 문맥상 정확한 시/도와 시/군/구를 찾으세요.
2. 대학명이나 특별한 랜드마크가 언급되면 해당 위치를 참고하세요:
   - KAIST, 한국과학기술원 → 대전광역시 유성구
   - 서울대학교 → 서울특별시 관악구
   - 연세대학교 → 서울특별시 서대문구
3. 위 목록에 없는 시도명이나 과거 행정구역명(예: 강원도→강원특별자치도, 전라북도→전북특별자치도)은 현재 명칭으로 변경하여 출력하세요.

출력 포맷:{{format_instructions}}

입력:{{query}}
"""
    
    response_schemas = [
        ResponseSchema(name="province", description="시/도 단위 지역 (예: 서울특별시, 경기도, 부산광역시 등) - 현재 유효한 명칭만 사용", type="string"),
        
        ResponseSchema(name="city", description="시/군/구 단위 지역 (예: 강남구, 수원시, 해운대구 등) - 해당 시/도에 실제 존재하는 명칭만 사용", type="string"),
        
        ResponseSchema(name="region", description="동/읍/면 단위 지역 (예: 역삼동, 장안면, 좌동 등) - 해당 시/군/구에 실제 존재하는 명칭만 사용", type="string"),
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate.from_template(
        template = enhanced_prompt,
        partial_variables={"format_instructions": format_instructions},
    )
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | model | output_parser
    
    return chain  

def weather_area_llm_chain():
    
    # 한국 지역 데이터 헬퍼 인스턴스 생성
    regions_helper = korea_regions_helper()
    
    # 유효한 지역명 목록 가져오기
    valid_provinces = regions_helper.get_valid_provinces()
    
    # 날씨 조회 전용 프롬프트 (더 상세한 가이드라인 포함)
    weather_prompt = f"""
입력한 문장에서 날씨 정보를 조회하고자 하는 한국의 지역을 정확히 추출하세요.
시/도 단위 지역과 시/군/구 단위 지역 그리고 동/읍/면 단위 지역을 각각 하나씩 선택하시오. 
추출할 수 없는 정보는 None을 출력하시오.
실제로 존재하지 않는 지역명은 반드시 None이라고 출력해야 함.

현재 유효한 시/도명 목록:
{', '.join(valid_provinces)}

중요한 지역명 매핑 규칙:
1. 문지동, 탑립동 → 대전광역시 유성구 (서울이 아님!)
2. 판교동 → 경기도 성남시 분당구
3. 역삼동, 삼성동, 청담동 → 서울특별시 강남구
4. 강남역 주변 → 서울특별시 강남구 역삼동
5. 홍대 → 서울특별시 마포구 서교동
6. 명동 → 서울특별시 중구 명동
7. 신촌 → 서울특별시 서대문구 창천동

주의사항:
1. 동명이 같더라도 반드시 문맥상 정확한 시/도와 시/군/구를 찾으세요.
2. 특별한 언급이 없으면 가장 일반적이고 알려진 지역으로 추정하되, 동명이 여러 곳에 있을 수 있으므로 주의하세요.
3. 대학명이 언급되면 해당 대학 위치를 참고하세요:
   - KAIST, 한국과학기술원 → 대전광역시 유성구
   - 서울대학교 → 서울특별시 관악구
   - 연세대학교 → 서울특별시 서대문구
4. ⚠️ 중요: 지역이 전혀 명시되지 않은 경우 (예: "오늘 날씨 어때?", "내일 비와?", "날씨 알려줘") 
   모든 필드를 None으로 출력하세요. 추정하지 마세요!

출력 포맷:{{format_instructions}}

입력:{{query}}
"""
    
    response_schemas = [
        ResponseSchema(name="province", description="시/도 단위 지역 (현재 유효한 법정 명칭만 사용)", type="string"),
        
        ResponseSchema(name="city", description="시/군/구 단위 지역 (해당 시/도에 실제 존재하는 명칭만 사용)", type="string"),
        
        ResponseSchema(name="region", description="동/읍/면 단위 지역 (해당 시/군/구에 실제 존재하는 명칭만 사용)", type="string"),
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate.from_template(
        template = weather_prompt,
        partial_variables={"format_instructions": format_instructions},
    )
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | model | output_parser
    
    return chain  

def summary_llm_chain():
    prompt = PromptTemplate.from_template(
        template = "다음 문단을 한국어로 아이에게 애기하듯이 요약해주세요: ~해요, ~어요 체를 써줘 \n\n{query}")
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = prompt | model
    
    return chain

def categorize_menu_llm(query):
    
    chain = categorize_llm_chain()
    
    return chain.invoke({"query": query})

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
        
        filepath = os.path.join(os.path.dirname(__file__), "xylist.xlsx")
        
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
            
            st.write(weather_text)
            
            return weather_text

        except requests.exceptions.RequestException as e:
            print(f"네트워크 오류: {e}")
            return f"죄송해요, 네트워크 문제로 날씨 정보를 가져올 수 없어요. 인터넷 연결을 확인하고 다시 시도해주세요."
            
        except Exception as e:
            print(f"날씨 데이터 처리 오류: {e}")
            return f"죄송해요, 날씨 데이터를 처리하는 중 문제가 발생했어요. 잠시 후 다시 시도해주세요."

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

class transport_infos: # 교통 정보 관련 추천 툴
    
    def __init__(self):
        pass
    
    def get_transport_info(self, query):
        pass

# """ Helper Classes """

class korea_regions_helper:
    """
    한국 법정동 코드를 기반으로 정확한 지역명을 검증하고 추천하는 헬퍼 클래스
    """
    
    def __init__(self):
        self.regions_df = None
        self.load_regions_data()
    
    def load_regions_data(self):
        """korea_regions.csv 파일을 로드합니다."""
        try:
            filepath = os.path.join(os.path.dirname(__file__), "korea_regions.csv")
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
    
    def get_valid_cities_for_province(self, province):
        """특정 시도에 속하는 유효한 시군구명 목록을 반환합니다."""
        if self.regions_df is None or not province:
            return []
        
        cities = self.regions_df[
            (self.regions_df['시도명'] == province) & 
            (self.regions_df['시군구명'] != '')
        ]['시군구명'].unique().tolist()
        
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

# """ Helper functions """
def setup_env():
    
    env_path = os.path.join(os.getcwd(), '../.env')

    if os.path.exists(env_path):
        
        load_dotenv(dotenv_path=env_path)
        
        print(f"Loaded environment variables from: \033[94m{env_path}\033[0m")
        
    else:
        print("\033[91mError: .env file not found. Please create one with your OPENAI_API_KEY.\033[0m")
        
        sys.exit(1)

def print_history():
    
    for msg in st.session_state["messages"]:
        
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    """
        대화 기록을 추가합니다
    """
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

def define_session_state():
    """
        세션 상태 변수를 정의합니다.
    """
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "categorize_chain" not in st.session_state:
        st.session_state["categorize_chain"] = categorize_llm_chain()
    
    if "weather_area" not in st.session_state:
        st.session_state["weather_area"] = weather_area_llm_chain()
        
    if "tavily_client" not in st.session_state:
        st.session_state["tavily_client"] = web_search()
        
    if "summary_chain" not in st.session_state:
        st.session_state["summary_chain"] = summary_llm_chain()
        
    if "chatbot_chain" not in st.session_state:
        st.session_state["chatbot_chain"] = chatbot_llm_chain()
        
    if "region_chain" not in st.session_state:
        st.session_state["region_chain"] = region_llm_chain()
        
    if "regions_helper" not in st.session_state:
        st.session_state["regions_helper"] = korea_regions_helper()
        
    if "weather_forecast_tool" not in st.session_state:
        st.session_state["weather_forecast_tool"] = weather_forecast()      
        
    if "place_recommand_tool" not in st.session_state:
        st.session_state["place_recommand_tool"] = place_recommand()
        
    if "transport_infos_tool" not in st.session_state:
        st.session_state["transport_infos_tool"] = transport_infos()
        
def main():

    setup_env()
    
    define_session_state()
    
    print_history()
    
    # 메인 로직
    if user_input := st.chat_input(): # 입력 받는 부분
        
        add_history("user", user_input)
        
        st.chat_message("user").write(user_input)
        
        with st.chat_message("assistant"):
            
            # 첫번째 LLM 카테고리 실행
            response = st.session_state["categorize_chain"].invoke({"query": user_input})
            
            print(f"\033[95m{'='*50}\033[0m")
            print(f"\033[96m 분류 결과: \033[93m{response['category']}\033[0m")
            print(f"\033[95m{'='*50}\033[0m")
            
            # Google Places API 활용
            if response["category"] == CATEGORIES[0]: # 맛집

                # 1. 지역 추출
                region_response = st.session_state["region_chain"].invoke(
                    {"query": user_input}
                )
                
                province = region_response.get('province')
                city = region_response.get('city')
                region = region_response.get('region')
                
                # 1.5. 지역명 유효성 검증 (경고만 표시, 검색은 계속 진행)
                validation_result = st.session_state["regions_helper"].validate_location(
                    province=province, city=city, region=region
                )
                
                if not validation_result["valid"] and (province or city):
                    print(f"WARNING: 맛집 검색 - 지역명 검증 실패: {validation_result}")
                
                # 2. 검색 쿼리 생성
                location_text = f"{province} {city} {region}" if province or city or region else ""
                
                # '맛집' 키워드가 명시되어 있지 않으면 추가
                if "맛집" not in user_input and "식당" not in user_input:
                    search_query = f"{location_text.strip()} 맛집, 한국"
                else:
                    search_query = f"{user_input.strip()}, 한국"
                
                # 3. 맛집 검색 실행
                restaurants = st.session_state["place_recommand_tool"].search_restaurants(search_query)
                
                context_for_chatbot = ""
                
                if restaurants:
                    
                    # 4. 검색 결과를 챗봇이 읽을 수 있는 컨텍스트로 포맷팅
                    context_for_chatbot += f"'{search_query}'에 대한 검색 결과입니다 (총 {len(restaurants)}개):\n\n"
                    
                    # 상위 5개 또는 10개만 추출하여 보여주는 것이 좋습니다. 여기서는 상위 5개로 제한합니다.
                    for i, place in enumerate(restaurants[:5]): 
                        name = place.get('displayName', {}).get('text', '이름 없음')
                        address = place.get('formattedAddress', '주소 정보 없음')
                        rating = place.get('rating', '평점 없음')
                        price_level = place.get('priceLevel', '가격대 정보 없음') # 예: PRICE_LEVEL_MODERATE (1-4)
                        reviews = place.get('reviews', []) # 리뷰 리스트 추출
                        
                        # 가격대 레벨을 한국어로 변환 (예시)
                        price_map = {
                            'PRICE_LEVEL_FREE': '무료',
                            'PRICE_LEVEL_VERY_INEXPENSIVE': '매우 저렴',
                            'PRICE_LEVEL_INEXPENSIVE': '저렴',
                            'PRICE_LEVEL_MODERATE': '적당함',
                            'PRICE_LEVEL_EXPENSIVE': '비쌈',
                            'PRICE_LEVEL_VERY_EXPENSIVE': '매우 비쌈'
                        }
                        price_str = price_map.get(price_level, '정보 없음')
                        
                        # 첫 번째 리뷰 텍스트 추출
                        first_review_text = ""
                        if reviews and reviews[0].get('text', {}).get('text'):
                             first_review_text = reviews[0]['text']['text'][:100] + "..." # 100자까지 잘라냄
                        
                        
                        context_for_chatbot += f"{i+1}. **{name}**\n"
                        context_for_chatbot += f"   - 주소: {address}\n"
                        context_for_chatbot += f"   - 평점: {rating}\n"
                        context_for_chatbot += f"   - 가격대: {price_str}\n"
                        if first_review_text:
                            context_for_chatbot += f"   - **최신 리뷰 요약**: {first_review_text}\n"
                        context_for_chatbot += "\n"
                        
                    if len(restaurants) > 5:
                        context_for_chatbot += f"...외 {len(restaurants) - 5}개 더 검색되었습니다.\n"
                        
                    # 5. 챗봇에게 컨텍스트와 사용자 입력 전달하여 최종 응답 생성
                    response_from_chatbot = st.session_state["chatbot_chain"].invoke({
                            "context": context_for_chatbot,
                            "user_input": user_input
                    })
                    
                    st.write(response_from_chatbot.content)
                    
                else:
                    # 검색 결과가 없을 때
                    error_msg = f"미안해요, '{search_query}'에 대한 맛집 정보를 찾지 못했어요. 다른 지역이나 키워드로 다시 알려줄래요?"
                    st.write(error_msg)
            
            # Google Places API 활용
            elif response["category"] == CATEGORIES[1]: # 관광지
                # 1. 지역 추출
                region_response = st.session_state["region_chain"].invoke(
                    {"query": user_input}
                )
                
                province = region_response.get('province')
                city = region_response.get('city')
                region = region_response.get('region')
                
                # 1.5. 지역명 유효성 검증 (경고만 표시, 검색은 계속 진행)
                validation_result = st.session_state["regions_helper"].validate_location(
                    province=province, city=city, region=region
                )
                
                if not validation_result["valid"] and (province or city):
                    print(f"WARNING: 관광지 검색 - 지역명 검증 실패: {validation_result}")
                
                # 2. 검색 쿼리 생성
                location_text = f"{province} {city} {region}" if province or city or region else ""
                
                # '관광지' 키워드가 명시되어 있지 않으면 추가
                if "관광지" not in user_input and "가볼 만한 곳" not in user_input and "볼거리" not in user_input:
                    search_query = f"{location_text.strip()} 관광지, 한국"
                else:
                    search_query = f"{user_input.strip()}, 한국"
                
                # 3. 관광지 검색 실행
                places = st.session_state["place_recommand_tool"].search_places(search_query)
                
                context_for_chatbot = ""
                
                if places:
                    
                    # 4. 검색 결과를 챗봇이 읽을 수 있는 컨텍스트로 포맷팅
                    context_for_chatbot += f"'{search_query}'에 대한 관광지 검색 결과입니다 (총 {len(places)}개):\n\n"
                    
                    # 상위 5개로 제한합니다.
                    for i, place in enumerate(places[:5]): 
                        name = place.get('displayName', {}).get('text', '이름 없음')
                        address = place.get('formattedAddress', '주소 정보 없음')
                        rating = place.get('rating', '평점 없음')
                        
                        reviews = place.get('reviews', []) # 리뷰 리스트 추출
                        
                        # 첫 번째 리뷰 텍스트 추출
                        first_review_text = ""
                        if reviews and reviews[0].get('text', {}).get('text'):
                             first_review_text = reviews[0]['text']['text'][:100] + "..." # 100자까지 잘라냄
                        
                        
                        context_for_chatbot += f"{i+1}. **{name}**\n"
                        context_for_chatbot += f"   - 주소: {address}\n"
                        context_for_chatbot += f"   - 평점: {rating}\n"
                        if first_review_text:
                            context_for_chatbot += f"   - **최신 리뷰 요약**: {first_review_text}\n"
                        context_for_chatbot += "\n"
                        
                    if len(places) > 5:
                        context_for_chatbot += f"...외 {len(places) - 5}개 더 검색되었습니다.\n"
                        
                    # 5. 챗봇에게 컨텍스트와 사용자 입력 전달하여 최종 응답 생성
                    response_from_chatbot = st.session_state["chatbot_chain"].invoke({
                            "context": context_for_chatbot,
                            "user_input": user_input
                    })
                    
                    st.write(response_from_chatbot.content)
                    
                else:
                    # 검색 결과가 없을 때
                    error_msg = f"미안해요, '{search_query}'에 대한 관광지 정보를 찾지 못했어요. 다른 지역이나 키워드로 다시 알려줄래요?"
                    st.write(error_msg)
            
            # DATA KR 동네예보 서비스 API 활용
            elif response["category"] == CATEGORIES[2]: # 날씨
                
                # 지역 추출
                location_response = st.session_state["weather_area"].invoke(
                    {"query": user_input}
                )
                
                province = location_response.get('province')
                city = location_response.get('city')
                region = location_response.get('region')
                
                # 지역명 유효성 검증
                validation_result = st.session_state["regions_helper"].validate_location(
                    province=province, city=city, region=region
                )
                
                # 지역이 전혀 명시되지 않은 경우 (모든 값이 None이거나 'None')
                if (not province or province == 'None') and (not city or city == 'None') and (not region or region == 'None'):
                    error_msg = """
                    🗺️ 어느 지역의 날씨를 알고 싶으신가요?
                    
                    예시로 이렇게 물어보세요:
                    • "서울 강남구 날씨 알려줘"
                    • "대전 유성구 문지동 날씨는?"
                    • "부산 해운대 날씨 어때?"
                    • "제주도 날씨 궁금해"
                    
                    지역을 구체적으로 말씀해주시면 정확한 날씨 정보를 드릴게요! 😊
                    """
                    st.write(error_msg)
                    print(f"INFO: 지역이 명시되지 않음 - province={province}, city={city}, region={region}")
                    return
                
                if not validation_result["valid"]:
                    # 유효하지 않은 지역명인 경우 사용자에게 알림
                    error_messages = []
                    suggestions_text = ""
                    
                    for field, message in validation_result["corrections"].items():
                        error_messages.append(message)
                    
                    if validation_result["suggestions"]:
                        suggestions_text = "\n\n💡 혹시 이런 지역을 찾으시나요?\n" + "\n".join([f"• {s}" for s in validation_result["suggestions"]])
                    
                    error_msg = f"죄송해요, 입력해주신 지역 정보를 정확히 찾지 못했어요:\n\n" + "\n".join([f"• {msg}" for msg in error_messages]) + suggestions_text + "\n\n정확한 지역명(시도, 시군구, 동)을 다시 말씀해 주세요!"
                    st.write(error_msg)
                    print(f"INFO: 지역명 검증 실패 - {validation_result}")
                else:
                    # 유효한 지역명인 경우 날씨 조회 진행
                    context_weather = st.session_state["weather_forecast_tool"].get_weather_forcast(
                        province, city, region
                    )
                    
                    if context_weather and not context_weather.startswith("날씨 조회 실패"):
                        response = st.session_state["chatbot_chain"].invoke({
                                "context": f"다음은 {province} {city} {region}의 날씨 정보입니다:\n\n{context_weather}\n\n위 정보를 바탕으로 사용자의 질의에 친절하게 설명해줘",
                                "user_input": user_input
                        })
                        
                        st.write(response.content)
                    else:
                        # 날씨 API 호출 실패
                        st.write("죄송해요, 현재 날씨 정보를 가져올 수 없어요. 잠시 후 다시 시도해주세요.")
                 
            # OK!          
            elif response["category"] == CATEGORIES[3]: # 검색
                
                try:
                    # Tavily 검색 API 호출
                    search_response = st.session_state["tavily_client"].search(user_input)

                    # 결과 포맷팅 시작
                    formatted_output = ""
                    
                    # LLM으로 답변 요약
                    if search_response.get('answer'):
                        try:
                            answer_obj = st.session_state["summary_chain"].invoke({"query": search_response['answer']})
                            answer_text = answer_obj.content if hasattr(answer_obj, 'content') else str(answer_obj)
                        
                        except Exception as summary_error:
                            print(f"요약 생성 중 오류: {summary_error}")
                            answer_text = search_response['answer']  # 원본 답변 사용
                    
                        formatted_output += f"💡 답변:\n"
                        formatted_output += f"> {answer_text}\n\n"
                        formatted_output += "-" * 40 + "\n"
                        
                    # 2. 개별 검색 결과 (Results)
                    if search_response.get('results'):
                        
                        for i, result in enumerate(search_response['results']):
                            title = result.get('title', '제목 없음')
                            url = result.get('url', 'URL 없음')
                            
                            formatted_output += f"\n -[{i+1}. {title}]**\n"
                            formatted_output += f" -- 출처: {url}\n"
                            
                    else:
                        formatted_output += "검색 결과를 찾지 못했습니다.\n"

                    formatted_output += "\n========================================\n"
                    
                    response = st.session_state["chatbot_chain"].invoke({
                        "context": f"다음은 검색 결과입니다:\n\n {formatted_output} \n\n 위 정보를 바탕으로 사용자의 질의에 친절하게 설명해줘",
                        "user_input": user_input
                    })
                    
                    st.write(response.content)
                    
                except Exception as e:
                    st.error(f"검색 중 오류가 발생했습니다: {e}")
                    print(f"오류 타입: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())
                    
                except Exception as e:
                    st.error(f"검색 중 오류가 발생했습니다: {e}")
                    print(f"오류 타입: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # OK!          
            elif response["category"] == CATEGORIES[4] or response["category"] == CATEGORIES[5]: # 현재 시간 또는 현재 날짜
                # 한국 시간(KST, UTC+9) 기준 현재 날짜와 시간 조회
                
                now_kst = datetime.now(timezone(timedelta(hours=9)))
                
                current_date = now_kst.strftime("%Y년 %m월 %d일")
                current_time = now_kst.strftime("%H시 %M분 %S초")

                response = st.session_state["chatbot_chain"].invoke({
                        "context": f"현재 날짜는 {current_date}이고, 현재 시간은 {current_time}입니다.",
                        "user_input": user_input
                })
                
                st.write(response.content)
            
            # 국토교통부_(TAGO)_버스도착정보 API 활용
            elif response["category"] == CATEGORIES[6]: # 교통편 조회
                pass
            
            
            # add_history("ai", str(response))
               
if __name__ == "__main__":
    
    main()