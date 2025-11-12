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

def weather_area_llm_chain():
    
    response_schemas = [
        ResponseSchema(name="province", description="시/도 단위 지역 (예: 서울특별시, 경기도, 부산광역시 등)", type="string"),
        
        ResponseSchema(name="city", description="시/군/구 단위 지역 (예: 강남구, 수원시, 해운대구 등)", type="string"),
        
        ResponseSchema(name="region", description="동/읍/면 단위 지역 (예: 역삼동, 장안면, 좌동 등)", type="string"),
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate.from_template(
        template = GET_PROVINCE_CITY_PROMPT,
        partial_variables={"format_instructions": format_instructions, "categories": CATEGORIES},
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
            
        # 지역명 필터링 (동/읍/면 단위로 검색하는 것이 가장 정확)
        query = self.xy_list[
            (self.xy_list['province'] == self.province) &
            (self.xy_list['city'] == self.city) &
            (self.xy_list['region'] == self.region)
        ]
        
        if not query.empty:
            # 첫 번째 일치하는 행의 데이터를 사용합니다.
            row = query.iloc[0]
            return {
                'nx': row['nx'],
                'ny': row['ny'],
                'lat': row['lat'],
                'lon': row['lon']
            }
        else:
            # 동/읍/면 단위에서 못 찾았을 경우 시/군/구 단위로 다시 검색 (예: 특정 동이 통합되었을 경우)
            query = self.xy_list[
                (self.xy_list['province'] == self.province) &
                (self.xy_list['city'] == self.city)
            ]
            
            if not query.empty:
                # 시/군/구의 대표 지점 (예: 첫 번째 행)의 좌표를 사용합니다.
                row = query.iloc[0]
                print(f"WARNING: '{self.region}'에 대한 정확한 좌표를 찾을 수 없어, '{self.city}'의 대표 좌표를 사용합니다.")
                return {
                    'nx': row['nx'],
                    'ny': row['ny'],
                    'lat': row['lat'],
                    'lon': row['lon']
                }
            
            print(f"ERROR: '{self.province} {self.city} {self.region}'에 해당하는 좌표를 찾을 수 없습니다.")
            
            return None

    def get_current_datetime(self):
        """
        현재 날짜와 시간을 'yyyyMMdd' 및 'HHMM' 형식으로 반환
        
        Returns:
            tuple: (date_str, time_str)
        """
        # 한국 표준시(KST, UTC+9)로 현재 시각을 얻고, 기준시는 2시간 전으로 설정
        now = datetime.now(timezone(timedelta(hours=9)))
        base_time = now - timedelta(hours=2)
        
        date_str = now.strftime("%Y%m%d")
        time_str = base_time.strftime("%H00")

        return date_str, time_str

    def get_weather_forcast(self, province, city, region):  
        
        self.set_location(province, city, region)
        
        coords = self.get_coordinates()
        
        date_str, time_str = self.get_current_datetime()
        
        if coords is None:
            st.error(f"날씨 조회 실패: 없는 지역 입니다, 지역을 다시 입력해주세요")
            return

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
                print(f"API 오류: {data.get('response', {}).get('header', {}).get('resultMsg')}")
                return
            
            items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            
            if not items:
                print("예보 데이터가 없습니다.")
                return
            
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
            
            return weather_text

        except requests.exceptions.RequestException as e:
            st.error(f"네트워크 오류 또는 API 호출 실패: {e}")
            
        except Exception as e:
            st.error(f"날씨 데이터를 처리하는 중 오류가 발생했습니다: {e}")

class place_recommand: # 맛집, 관광지 등의 맛집 추천 툴
    
    def __init__(self):
        pass
    
    def recommend(self, category, location):
        pass

class transport_infos: # 교통 정보 관련 추천 툴
    
    def __init__(self):
        pass
    
    def get_transport_info(self, query):
        pass
    
# """ Helper functions """"
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
                pass
            
            # Google Places API 활용
            elif response["category"] == CATEGORIES[1]: # 관광지
                pass
            
            # DATA KR 동네예보 서비스 API 활용
            elif response["category"] == CATEGORIES[2]: # 날씨
                
                response = st.session_state["weather_area"].invoke(
                    {"query": user_input}
                )
                
                context_weather = st.session_state["weather_forecast_tool"].get_weather_forcast(
                    response['province'],
                    response['city'],
                    response['region'],
                )
                
                response = st.session_state["chatbot_chain"].invoke({
                        "context": f"다음은 {response['province']} {response['city']} {response['region']}의 날씨 정보입니다: \n\n 위 정보를 바탕으로 사용자의 질의에 친절하게 설명해줘",
                        "user_input": user_input
                })
                
                st.write(response.content)
                 
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

                st.write(f"📅 현재 날짜: {current_date}")
                st.write(f"🕐 현재 시간: {current_time}")
            
            # 국토교통부_(TAGO)_버스도착정보 API 활용
            elif response["category"] == CATEGORIES[6]: # 교통편 조회
                pass
            
            
            # add_history("ai", str(response))
               
if __name__ == "__main__":
    
    main()