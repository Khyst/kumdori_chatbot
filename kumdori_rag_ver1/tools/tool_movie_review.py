import requests
import pandas as pd

def get_movie_list():
    
    TMDB_API_KEY = "f62d3429dcfdafe78bd08bfd9e30778e"

    # 가져올 페이지 수 (예: 1부터 5페이지까지)
    TOTAL_PAGES_TO_FETCH = 5 

    # 데이터를 저장할 빈 리스트
    all_movies_data = []

    print(f"--- TMDB API에서 {TOTAL_PAGES_TO_FETCH} 페이지의 인기 영화 데이터를 가져오는 중 ---")

    # 지정된 페이지 수만큼 반복하여 데이터 가져오기
    for page in range(1, TOTAL_PAGES_TO_FETCH + 1):
        # TMDB 인기 영화 API 엔드포인트
        url = f"https://api.themoviedb.org/3/movie/popular"
        
        # 요청에 필요한 파라미터 설정
        params = {
            "api_key": TMDB_API_KEY,
            "language": "ko-KR",  # 언어 설정 (한국어)
            "page": page           # 현재 페이지 번호
        }

        try:
            # API 요청 보내기
            response = requests.get(url, params=params)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

            # JSON 응답을 Python 딕셔너리로 변환
            data = response.json()
            
            # 'results' 키에 실제 영화 목록이 포함되어 있습니다.
            movies_on_page = data.get("results", [])
            
            if not movies_on_page:
                print(f"페이지 {page}: 데이터가 없습니다. 중단합니다.")
                break
                
            # 현재 페이지의 영화 데이터를 전체 리스트에 추가
            all_movies_data.extend(movies_on_page)
            print(f"페이지 {page} 로드 완료. 현재까지 총 {len(all_movies_data)}개의 영화 데이터 누적.")

        except requests.exceptions.RequestException as e:
            print(f"API 요청 중 오류 발생 (페이지 {page}): {e}")
            break

    # 전체 데이터를 사용하여 Pandas DataFrame 생성
    if all_movies_data:
        df = pd.DataFrame(all_movies_data)
        print("\n--- 데이터 가져오기 완료 ---")
        print(f"최종 DataFrame에 저장된 총 영화 수: {len(df)}")        
        return df

    else:
        print("\n--- 데이터 가져오기 실패. DataFrame을 생성하지 못했습니다. ---")
        
if __name__ == "__main__":
    print(get_movie_list())
    
