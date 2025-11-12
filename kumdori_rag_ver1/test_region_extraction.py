#!/usr/bin/env python3
"""
지역 추출 및 검증 시스템 테스트 스크립트
"""

import sys
import os
import pandas as pd

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the korea_regions_helper from the main file
from kumdori_chatbot_node import korea_regions_helper

def test_location_extraction():
    """지역 추출 및 검증 시스템 테스트"""
    
    print("🔍 한국 지역명 추출 및 검증 시스템 테스트")
    print("=" * 60)
    
    # Initialize the helper
    regions_helper = korea_regions_helper()
    
    # Test cases with common location extraction scenarios
    test_cases = [
        # Old vs New province names
        {"province": "강원도", "city": "춘천시", "region": None, "description": "구 도명 → 신 도명"},
        {"province": "전라북도", "city": "전주시", "region": None, "description": "구 도명 → 신 도명"},
        
        # City name variations
        {"province": "부산시", "city": "해운대구", "region": None, "description": "부산시 → 부산광역시"},
        
        # Merged city issues
        {"province": "경상남도", "city": "진해시", "region": None, "description": "통합된 도시 (진해시)"},
        {"province": "경상남도", "city": "마산시", "region": None, "description": "통합된 도시 (마산시)"},
        
        # Valid locations
        {"province": "서울특별시", "city": "강남구", "region": "역삼동", "description": "완전히 유효한 지역"},
        {"province": "경기도", "city": "수원시", "region": None, "description": "유효한 시도/시군구"},
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['description']}")
        print(f"   입력: {case['province']} {case['city']} {case['region']}")
        print("-" * 50)
        
        result = regions_helper.validate_location(
            province=case['province'],
            city=case['city'],
            region=case['region']
        )
        
        if result["valid"]:
            print("   ✅ 유효한 지역명")
        else:
            print("   ❌ 유효하지 않은 지역명")
            for field, message in result["corrections"].items():
                print(f"      - {message}")
            
            if result["suggestions"]:
                print("   💡 추천 수정사항:")
                for suggestion in result["suggestions"]:
                    print(f"      - {suggestion}")
    
    # Display valid provinces
    print(f"\n📍 현재 유효한 시도명 목록 ({len(regions_helper.get_valid_provinces())}개):")
    for province in regions_helper.get_valid_provinces():
        print(f"   - {province}")
    
    # Show examples of cities for specific provinces
    print(f"\n🏙️  경상남도 시군구 예시:")
    gyeongnam_cities = regions_helper.get_valid_cities_for_province("경상남도")
    changwon_related = [city for city in gyeongnam_cities if "창원시" in city]
    print(f"   창원시 관련 ({len(changwon_related)}개): {', '.join(changwon_related)}")
    
    regular_cities = [city for city in gyeongnam_cities if "창원시" not in city][:10]
    print(f"   기타 시군 (처음 10개): {', '.join(regular_cities)}")

if __name__ == "__main__":
    test_location_extraction()