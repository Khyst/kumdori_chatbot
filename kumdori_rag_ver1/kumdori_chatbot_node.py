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

# """ 내부 Tools 모듈 임포트 """
from verificators.korea_regions_verificator import korea_regions_verificator

from tools.tool_place_recommand import place_recommand
from tools.tool_weather_forcast import weather_forecast
from tools.tool_web_search import web_search
from tools.tool_transport_infos import transport_infos

# """ Streamlit GUI 라이브러리 """
import streamlit as st

# """ 전역 변수 및 상수 정의 """
PERSONA_INSTRUCTIONS = """당신은 한국어에 능통한 친절한 챗봇입니다. 사용자가 질문하면 사용자의 질문에 대한 답변을 제공해야 합니다. 한국어로 아이에게 애기하듯이 말해주세요, 추후 목소리로 말할 수 있는 기능에 대비하여 기호는 사용하지 말고 말로 부드럽게 해야합니다. 없는 정보는 애기하지 말고, 모르면 모른다고 말하세요. 잘못된 정보를 제시하면 $100의 벌금을 부과할 겁니다, 검색한 정보에 대해서는 관련 링크를 같이 제시하면 좋아, 최종 답변은 사람에게 말하듯 하는 답변이어야 돼."""
PERSONA_CHARACTER = """ 당신은 꿈돌이 로봇으로, 항상 밝고 긍정적인 태도로 대화에 임하며, 사용자가 편안함을 느낄 수 있도록 친근하게 대화합니다. """
PERSONA_PROMPT = PERSONA_INSTRUCTIONS + "\n\n\n\n" + PERSONA_CHARACTER + "\n\n\n\n" + "아래와 같은 어투를 사용해서 답변 해, 반드시! 예시) 안녕! 나는 꿈돌이 로봇이야. 너와 이야기하는 걸 정말 좋아해. 궁금한 게 있으면 언제든지 물어봐. 함께 재미있는 이야기 나눠보자. 안녕하세요, 와 같은 존댓말 보다는 친근한 어투를 써줘"
CATEGORIZE_PROMPT = "이전 대화를 참고하여 입력한 문장을 분석하고, 다음의 카테고리 리스트에서 가장 가까운 카테고리 하나를 선택하시오.\n\n이전 대화:\n{chat_history}\n\n카테고리 리스트: {categories}\n출력 포맷:{format_instructions} \n\n입력:{query}"
GET_PROVINCE_CITY_PROMPT = "입력한 문장을 분석하여, 한국의 시/도 단위 지역과 시/군/구 단위 지역 그리고 동/읍/면 단위 지역을 각각 하나씩 선택하시오. 둘 중 하나라도 추출할 수 없다면 None을 출력하시오. 실제로 존재하지 않는 지역명은 반드시 None이라고 출력해야 함 \n 출력 포맷:{format_instructions} \n\n 입력:{query}"
CATEGORIES = ["맛집", "관광지", "날씨", "검색", "현재 시간", "현재 날짜", "교통"]
CURRENT_LOCATION="대전광역시 유성구 탑립동"

st.title("💬")

# 도움말 표시
st.markdown("""
<div style="background-color: #2c3e50; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <p style="margin-top: 0; color: #ecf0f1; font-weight: bold;">💡 예를 들면, 이런 것들을 도와드릴 수 있어요!</p>
    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
        <span style="background-color: #ff6b6b; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🍽️ 근처 맛집</span>
        <span style="background-color: #4ecdc4; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🏛️ 근처 가볼만한 곳</span>
        <span style="background-color: #45b7d1; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🌤️ 날씨</span>
        <span style="background-color: #96ceb4; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🔍 실시간 검색</span>
        <span style="background-color: #ffeaa7; color: #2d3436; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🕐 현재 시간</span>
        <span style="background-color: #fab1a0; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">📅 현재 날짜</span>
        <span style="background-color: #a29bfe; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">🚌 교통 정보 (개발중) </span>
        <span style="background-color: #fd79a8; color: white; padding: 5px 12px; border-radius: 15px; font-size: 14px; font-weight: bold;">💬 일상 대화</span>
    </div>
</div>
""", unsafe_allow_html=True)

# """ 각종 역할을 가지고 있는 LLM 체인들 """

def chatbot_llm_chain():
    prompt = PromptTemplate.from_template(
        template = PERSONA_PROMPT + "\n\n\n이전 대화 내역:\n{chat_history}\n\n\n 관련 정보: {context} \n\n\n 사용자 요청: {user_input} \n 꿈돌이 로봇:"
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

def summary_llm_chain():
    prompt = PromptTemplate.from_template(
        template = "다음 문단을 한국어로 아이에게 애기하듯이 요약해주세요: ~해요, ~어요 체를 써줘 \n\n{query}")
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    chain = prompt | model
    
    return chain

def categorize_menu_llm(query):
    
    chain = categorize_llm_chain()
    
    return chain.invoke({"query": query})

# """ Helper functions """

def extract_keywords_from_query(query):
    """
    사용자 쿼리에서 장소 특성 키워드를 추출합니다.
    """
    keywords = {
        "parking": ["주차", "주차장", "주차 공간", "주차가능"],
        "atmosphere": ["분위기", "인테리어", "깔끔", "예쁘", "감성", "무드"],
        "portion": ["양", "푸짐", "많", "넉넉"],
        "value": ["가성비", "저렴", "싸", "가격", "합리적"],
        "service": ["서비스", "친절", "직원"],
        "taste": ["맛", "맛있", "맛집", "존맛", "JMT"],
        "quiet": ["조용", "한적", "여유"],
        "view": ["뷰", "전망", "풍경", "야경"],
        "kids": ["아이", "어린이", "키즈", "가족"],
        "date": ["데이트", "커플", "연인"],
        "group": ["단체", "모임", "회식"],
        "clean": ["청결", "위생", "깨끗"],
        "photo": ["사진", "인스타", "감성샷", "포토"],
        "accessible": ["접근성", "가깝", "역 근처", "찾기 쉬운"]
    }
    
    found_keywords = []
    query_lower = query.lower()
    
    for category, keyword_list in keywords.items():
        for keyword in keyword_list:
            if keyword in query:
                found_keywords.append(keyword)
                break
    
    return found_keywords

def filter_places_by_keywords(places, keywords):
    """
    키워드를 기반으로 장소를 필터링하고 점수를 매깁니다.
    """
    if not keywords or not places:
        return places
    
    scored_places = []
    
    for place in places:
        score = 0
        # 리뷰에서 키워드 매칭
        reviews = place.get('reviews', [])
        
        for review in reviews:
            review_text = review.get('text', {}).get('text', '').lower()
            for keyword in keywords:
                if keyword.lower() in review_text:
                    score += 1
        
        # 장소 이름, 설명에서도 매칭
        name = place.get('displayName', {}).get('text', '').lower()
        for keyword in keywords:
            if keyword.lower() in name:
                score += 2  # 이름에 있으면 가중치 더 높게
        
        scored_places.append({
            'place': place,
            'score': score
        })
    
    # 점수순으로 정렬
    scored_places.sort(key=lambda x: x['score'], reverse=True)
    
    # 원본 place 객체만 반환
    return [item['place'] for item in scored_places]

def normalize_province_name(province_name):
    """
    과거 행정구역명을 현재 명칭으로 변환하는 함수
    """
    province_mappings = {
        "대전직할시": "대전광역시",
        "대구직할시": "대구광역시",
        "부산직할시": "부산광역시",
        "인천직할시": "인천광역시",
        "광주직할시": "광주광역시",
        "울산직할시": "울산광역시",
        "강원도": "강원특별자치도",
        "전라북도": "전북특별자치도",
        "전북도": "전북특별자치도",
        "제주도": "제주특별자치도"
    }
    
    return province_mappings.get(province_name, province_name)

def extract_region_from_text(text):
    """
    텍스트에서 지역 정보를 추출하는 내부 헬퍼 함수
    """
    if not text or text.strip() == "":
        return {"province": None, "city": None, "region": None}
    
    text_striped = text.strip()
    
    # 변수 초기화
    province = None
    city = None
    region = None
        
    reigion_verificator = korea_regions_verificator()
    
    # 1. 시/도 추출
    valid_provinces = reigion_verificator.get_valid_provinces()
    valid_provinces_sorted = sorted(valid_provinces, key=len, reverse=True)
    
    for elem in valid_provinces_sorted:
        if elem in text_striped:
            province = elem
            text_striped = text_striped.replace(elem, "").strip()
            break
    
    # 2. 시/군/구 추출
    if province:
        valid_cities = reigion_verificator.get_valid_cities_for_province(province)
    else:
        valid_cities = reigion_verificator.get_all_cities()
    
    valid_cities_sorted = sorted(valid_cities, key=len, reverse=True)
    
    for elem in valid_cities_sorted:
        if elem in text_striped:
            city = elem
            text_striped = text_striped.replace(elem, "").strip()
            
            if not province:
                province = reigion_verificator.get_province_for_city(city)
            break
    
    # 3. 동/읍/면 추출
    if province and city:
        valid_regions = reigion_verificator.get_valid_regions_for_city(province, city)
        
    elif city:
        valid_regions = reigion_verificator.get_all_regions_for_city(city)
        
    else:
        valid_regions = reigion_verificator.get_all_regions()
    
    valid_regions_sorted = sorted(valid_regions, key=len, reverse=True)
    
    for elem in valid_regions_sorted:
        if elem in text_striped:
            region = elem
            text_striped = text_striped.replace(elem, "").strip()
            
            if not city:
                location_info = reigion_verificator.get_location_for_region(region)
                if location_info:
                    province = location_info.get('province')
                    city = location_info.get('city')
            break
    
    return {
        "province": province,
        "city": city,
        "region": region
    }

def region_keyword_extractor(query):
    
    if query is None or query.strip() == "":
        return {"province": None, "city": None, "region": None}
    
    query_striped = query.strip()
    
    # 변수 초기화
    province = None
    city = None
    region = None
    
    # "여기", "이곳", "현재 위치", "우리 동네" 등의 키워드가 있으면 현재 위치 사용
    current_location_keywords = ["여기", "이곳", "현재 위치", "우리 동네", "이 근처"]
    use_current_location = any(keyword in query_striped for keyword in current_location_keywords)
    
    if use_current_location:
        # CURRENT_LOCATION에서 지역 정보 추출
        print(f"DEBUG: 현재 위치 키워드 감지 - {CURRENT_LOCATION} 사용")
        current_location_response = extract_region_from_text(CURRENT_LOCATION)
        print(f"DEBUG: 현재 위치에서 추출된 지역 - 시/도: {current_location_response['province']}, 시/군/구: {current_location_response['city']}, 동/읍/면: {current_location_response['region']}")
        return current_location_response
        
    reigion_verificator = korea_regions_verificator()
    
    # 1. 시/도 추출
    valid_provinces = reigion_verificator.get_valid_provinces()
    # 긴 이름부터 검색 (예: "경상남도"가 "경상"보다 먼저)
    valid_provinces_sorted = sorted(valid_provinces, key=len, reverse=True)
    
    for elem in valid_provinces_sorted:
        if elem in query_striped:
            province = elem
            query_striped = query_striped.replace(elem, "").strip()
            break
    
    # 2. 시/군/구 추출
    if province:
        # 시/도가 있으면 해당 시/도의 시/군/구만 검색
        valid_cities = reigion_verificator.get_valid_cities_for_province(province)
    else:
        # 시/도가 없으면 모든 시/군/구 검색
        valid_cities = reigion_verificator.get_all_cities()
    
    # 긴 이름부터 검색
    valid_cities_sorted = sorted(valid_cities, key=len, reverse=True)
    
    for elem in valid_cities_sorted:
        if elem in query_striped:
            city = elem
            query_striped = query_striped.replace(elem, "").strip()
            
            # city를 찾았는데 province가 없으면 역으로 province 찾기
            if not province:
                province = reigion_verificator.get_province_for_city(city)
            break
    
    # 3. 동/읍/면 추출
    if province and city:
        # 시/도와 시/군/구가 있으면 해당 지역의 동/읍/면만 검색
        valid_regions = reigion_verificator.get_valid_regions_for_city(province, city)
    elif city:
        # 시/군/구만 있으면 해당 시/군/구의 모든 동/읍/면 검색
        valid_regions = reigion_verificator.get_all_regions_for_city(city)
    else:
        # 아무것도 없으면 모든 동/읍/면 검색
        valid_regions = reigion_verificator.get_all_regions()
    
    # 긴 이름부터 검색 (중요! "송강동"이 "강동"보다 먼저 매칭되도록)
    valid_regions_sorted = sorted(valid_regions, key=len, reverse=True)
    
    for elem in valid_regions_sorted:
        if elem in query_striped:
            region = elem
            query_striped = query_striped.replace(elem, "").strip()
            
            # region을 찾았는데 상위 정보가 없으면 역으로 찾기
            if not city:
                location_info = reigion_verificator.get_location_for_region(region)
                if location_info:
                    province = location_info.get('province')
                    city = location_info.get('city')
            break
    
    # 과거 행정구역명을 현재 명칭으로 변환
    if province:
        province = normalize_province_name(province)

    print(f"DEBUG: 추출된 지역 - 시/도: {province}, 시/군/구: {city}, 동/읍/면: {region}")
        
    return {
        "province": province,
        "city": city,
        "region": region
    }

def setup_env():
    """ 
        .env 파일에서 환경 변수를 로드합니다.
    """
    env_path = os.path.join(os.getcwd(), '../.env')

    if os.path.exists(env_path):
        
        load_dotenv(dotenv_path=env_path)
        
        print(f"Loaded environment variables from: \033[94m{env_path}\033[0m")
        
    else:
        print("\033[91mError: .env file not found. Please create one with your OPENAI_API_KEY.\033[0m")
        
        sys.exit(1)

def print_history():
    """ 
        대화 기록을 출력합니다.
    """
    for msg in st.session_state["messages"]:
        
        st.chat_message(msg.role).write(msg.content)

def add_history(role, content):
    """
        대화 기록을 추가합니다
    """
    st.session_state["messages"].append(ChatMessage(role=role, content=content))

def get_chat_history_text(max_messages=5):
    """
        최근 대화 기록을 텍스트로 변환합니다.
        max_messages: 포함할 최대 메시지 수 (기본 5개, 즉 최근 5턴의 대화)
    """
    if not st.session_state.get("messages"):
        return "이전 대화 없음"
    
    recent_messages = st.session_state["messages"][-max_messages*2:] if len(st.session_state["messages"]) > max_messages*2 else st.session_state["messages"]
    
    history_text = ""
    
    for msg in recent_messages:
        role_name = "사용자" if msg.role == "user" else "꿈돌이"
        history_text += f"{role_name}: {msg.content}\n"
    
    return history_text.strip()

def define_session_state():
    """
        세션 상태 변수를 정의합니다.
    """
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    if "categorize_chain" not in st.session_state:
        st.session_state["categorize_chain"] = categorize_llm_chain()
    
    if "region_extractor" not in st.session_state:
        st.session_state["region_extractor"] = region_keyword_extractor
        
    if "tavily_client" not in st.session_state:
        st.session_state["tavily_client"] = web_search()
        
    if "summary_chain" not in st.session_state:
        st.session_state["summary_chain"] = summary_llm_chain()
        
    if "chatbot_chain" not in st.session_state:
        st.session_state["chatbot_chain"] = chatbot_llm_chain()
        
    if "regions_verificator" not in st.session_state:
        st.session_state["regions_verificator"] = korea_regions_verificator()
        
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
            
            # 대화 히스토리 가져오기
            chat_history = get_chat_history_text()
            
            # 0. 카테고리 추출 (히스토리 포함)
            response = st.session_state["categorize_chain"].invoke({
                "query": user_input,
                "chat_history": chat_history
            })
            
            print(f"\033[95m{'='*50}\033[0m")
            print(f"\033[96m 분류 결과: \033[93m{response['category']}\033[0m")
            print(f"\033[95m{'='*50}\033[0m")
            
            # Google Places API 활용
            if response["category"] == CATEGORIES[0]: # 맛집

                # 1. 지역 추출
                region_response = st.session_state["region_extractor"](user_input)
                
                province = region_response.get('province')
                city = region_response.get('city')
                region = region_response.get('region')
                
                # 2. 검색 쿼리 생성
                location_text = f"{province} {city} {region}" if province or city or region else ""
                
                # '맛집' 키워드가 명시되어 있지 않으면 추가
                if "맛집" not in user_input and "식당" not in user_input:
                    search_query = f"{location_text.strip()} 맛집, 한국"
                else:
                    search_query = f"{user_input.strip()}, 한국"
                
                # 2.5. 사용자 쿼리에서 특성 키워드 추출
                feature_keywords = extract_keywords_from_query(user_input)
                if feature_keywords:
                    print(f"DEBUG: 추출된 특성 키워드 - {feature_keywords}")
                
                # 3. 맛집 검색 실행
                restaurants = st.session_state["place_recommand_tool"].search_restaurants(search_query)
                
                # 3.5. 키워드로 필터링 및 정렬
                if restaurants and feature_keywords:
                    print(f"DEBUG: 키워드 기반 필터링 시작 - 원본 {len(restaurants)}개")
                    restaurants = filter_places_by_keywords(restaurants, feature_keywords)
                    print(f"DEBUG: 필터링 완료 - 정렬된 {len(restaurants)}개")
                
                context_for_chatbot = ""
                
                if restaurants:
                    
                    # 4. 검색 결과를 챗봇이 읽을 수 있는 컨텍스트로 포맷팅
                    context_for_chatbot += f"'{search_query}'에 대한 검색 결과입니다 (총 {len(restaurants)}개):\n"
                    if feature_keywords:
                        context_for_chatbot += f"✨ 특별히 '{', '.join(feature_keywords)}' 키워드에 맞춰 정렬되었습니다.\n\n"
                    else:
                        context_for_chatbot += "\n"
                    
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
                        
                        # 키워드 매칭된 리뷰 찾기
                        matched_reviews = []
                        if feature_keywords:
                            for review in reviews:
                                review_text = review.get('text', {}).get('text', '')
                                for keyword in feature_keywords:
                                    if keyword in review_text:
                                        matched_reviews.append(review_text[:100] + "...")
                                        break
                        
                        # 첫 번째 리뷰 텍스트 추출
                        first_review_text = ""
                        if matched_reviews:
                            first_review_text = matched_reviews[0]
                        elif reviews and reviews[0].get('text', {}).get('text'):
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
                            "chat_history": chat_history,
                            "context": context_for_chatbot,
                            "user_input": user_input
                    })
                    
                    st.write(response_from_chatbot.content)
                    add_history("assistant", response_from_chatbot.content)
                    
                else:
                    # 검색 결과가 없을 때
                    error_msg = f"미안해요, '{search_query}'에 대한 맛집 정보를 찾지 못했어요. 다른 지역이나 키워드로 다시 알려줄래요?"
                    st.write(error_msg)
                    add_history("assistant", error_msg)
            
            # Google Places API 활용
            elif response["category"] == CATEGORIES[1]: # 관광지
                
                # 1. 지역 추출
                location_response = st.session_state["region_extractor"](user_input)
                
                province = location_response.get('province')
                city = location_response.get('city')
                region = location_response.get('region')
                
                # 2. 검색 쿼리 생성
                location_text = f"{province} {city} {region}" if province or city or region else ""
                
                # '관광지' 키워드가 명시되어 있지 않으면 추가
                if "관광지" not in user_input and "가볼 만한 곳" not in user_input and "볼거리" not in user_input:
                    search_query = f"{location_text.strip()} 가볼 만한 곳, 한국"
                else:
                    search_query = f"{user_input.strip()}, 한국"
                
                # 2.5. 사용자 쿼리에서 특성 키워드 추출
                feature_keywords = extract_keywords_from_query(user_input)
                if feature_keywords:
                    print(f"DEBUG: 추출된 특성 키워드 - {feature_keywords}")
                
                # 3. 관광지 검색 실행
                places = st.session_state["place_recommand_tool"].search_places(search_query)
                
                # 3.5. 키워드로 필터링 및 정렬
                if places and feature_keywords:
                    print(f"DEBUG: 키워드 기반 필터링 시작 - 원본 {len(places)}개")
                    places = filter_places_by_keywords(places, feature_keywords)
                    print(f"DEBUG: 필터링 완료 - 정렬된 {len(places)}개")
                
                context_for_chatbot = ""
                
                if places:
                    
                    # 4. 검색 결과를 챗봇이 읽을 수 있는 컨텍스트로 포맷팅
                    context_for_chatbot += f"'{search_query}'에 대한 관광지 검색 결과입니다 (총 {len(places)}개):\n"
                    if feature_keywords:
                        context_for_chatbot += f"✨ 특별히 '{', '.join(feature_keywords)}' 키워드에 맞춰 정렬되었습니다.\n\n"
                    else:
                        context_for_chatbot += "\n"
                    
                    # 상위 5개로 제한합니다.
                    for i, place in enumerate(places[:5]): 
                        name = place.get('displayName', {}).get('text', '이름 없음')
                        address = place.get('formattedAddress', '주소 정보 없음')
                        rating = place.get('rating', '평점 없음')
                        
                        reviews = place.get('reviews', []) # 리뷰 리스트 추출
                        
                        # 키워드 매칭된 리뷰 찾기
                        matched_reviews = []
                        if feature_keywords:
                            for review in reviews:
                                review_text = review.get('text', {}).get('text', '')
                                for keyword in feature_keywords:
                                    if keyword in review_text:
                                        matched_reviews.append(review_text[:100] + "...")
                                        break
                        
                        # 첫 번째 리뷰 텍스트 추출
                        first_review_text = ""
                        if matched_reviews:
                            first_review_text = matched_reviews[0]
                        elif reviews and reviews[0].get('text', {}).get('text'):
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
                            "chat_history": chat_history,
                            "context": context_for_chatbot,
                            "user_input": user_input
                    })
                    
                    st.write(response_from_chatbot.content)
                    add_history("assistant", response_from_chatbot.content)
                    
                else:
                    # 검색 결과가 없을 때
                    error_msg = f"미안해요, '{search_query}'에 대한 관광지 정보를 찾지 못했어요. 다른 지역이나 키워드로 다시 알려줄래요?"
                    st.write(error_msg)
                    add_history("assistant", error_msg)
            
            # DATA KR 동네예보 서비스 API 활용
            elif response["category"] == CATEGORIES[2]: # 날씨
                
                location_response = st.session_state["region_extractor"](user_input)
                
                province = location_response.get('province')
                city = location_response.get('city')
                region = location_response.get('region')
                
                print(f"DEBUG: 추출된 지역 - 시/도: {province}, 시/군/구: {city}, 동/읍/면: {region}")
                
                # 지역이 전혀 명시되지 않은 경우 현재 위치 사용
                if (not province or province == 'None') and (not city or city == 'None') and (not region or region == 'None'):
                    print(f"DEBUG: 지역 미명시 - 현재 위치({CURRENT_LOCATION}) 사용")
                    current_location_info = extract_region_from_text(CURRENT_LOCATION)
                    province = current_location_info.get('province')
                    city = current_location_info.get('city')
                    region = current_location_info.get('region')
                    print(f"DEBUG: 현재 위치에서 추출 - 시/도: {province}, 시/군/구: {city}, 동/읍/면: {region}")
                
                validation_result = st.session_state["regions_verificator"].validate_location(
                    province=province, city=city, region=region
                )
                
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
                                "chat_history": chat_history,
                                "context": f"다음은 {province} {city} {region}의 날씨 정보입니다:\n\n{context_weather}\n\n위 정보를 바탕으로 사용자의 질의에 친절하게 설명해줘",
                                "user_input": user_input
                        })
                        
                        st.write(response.content)
                        add_history("assistant", response.content)
                    
                    else:
                        # 날씨 API 호출 실패
                        error_msg = "죄송해요, 현재 날씨 정보를 가져올 수 없어요. 잠시 후 다시 시도해주세요."
                        st.write(error_msg)
                        add_history("assistant", error_msg)
                 
            # Tavily 검색 API 활용
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
                        "chat_history": chat_history,
                        "context": f"다음은 검색 결과입니다:\n\n {formatted_output} \n\n 위 정보를 바탕으로 사용자의 질의에 친절하게 설명해줘",
                        "user_input": user_input
                    })
                    
                    st.write(response.content)
                    add_history("assistant", response.content)
                    
                except Exception as e:
                    error_msg = f"검색 중 오류가 발생했습니다: {e}"
                    st.error(error_msg)
                    add_history("assistant", error_msg)
                    print(f"오류 타입: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())
                    
                except Exception as e:
                    st.error(f"검색 중 오류가 발생했습니다: {e}")
                    print(f"오류 타입: {type(e).__name__}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # 기본 파이썬 datetime 모듈 활용
            elif response["category"] == CATEGORIES[4] or response["category"] == CATEGORIES[5]: # 현재 시간 또는 현재 날짜
                # 한국 시간(KST, UTC+9) 기준 현재 날짜와 시간 조회
                
                now_kst = datetime.now(timezone(timedelta(hours=9)))
                
                current_date = now_kst.strftime("%Y년 %m월 %d일")
                current_time = now_kst.strftime("%H시 %M분 %S초")

                response = st.session_state["chatbot_chain"].invoke({
                        "chat_history": chat_history,
                        "context": f"현재 날짜는 {current_date}이고, 현재 시간은 {current_time}입니다.",
                        "user_input": user_input
                })
                
                st.write(response.content)
                add_history("assistant", response.content)
            
            # 국토교통부_(TAGO)_버스도착정보 API 활용 (X)
            elif response["category"] == CATEGORIES[6]: # 교통편 조회
                info_msg = "죄송해요, 교통편 조회 기능은 아직 준비 중이에요."
                st.write(info_msg)
                add_history("assistant", info_msg)
               
if __name__ == "__main__":
    
    main()