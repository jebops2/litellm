#!/usr/bin/env python3
"""
Vertex AI Veo Video Generation - 간단한 예제

OpenAI 라이브러리를 사용하여 LiteLLM Proxy를 통해 Vertex AI Veo 비디오를 생성합니다.

사용 전 확인사항:
1. LiteLLM Proxy 서버가 실행 중이어야 합니다 (기본: http://localhost:4000)
2. 필요한 패키지 설치: pip install openai
"""

import os
import sys
from openai import OpenAI


def main():
    """메인 함수"""
    # 환경 변수에서 설정 읽기 (선택사항)
    proxy_url = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    api_key = os.getenv("LITELLM_API_KEY", "sk-1234")
    
    print(f"🔗 Proxy URL: {proxy_url}")
    print(f"🔑 API Key: {api_key[:10]}...")
    print()
    
    # OpenAI 클라이언트 초기화 (LiteLLM Proxy 사용)
    client = OpenAI(
        api_key=api_key,
        base_url=f"{proxy_url.rstrip('/')}/v1",
    )
    
    try:
        # 비디오 생성
        print("🎬 비디오 생성 중...")
        video = client.videos.create(
            model="veo3-0",
            prompt="A beautiful sunset over the ocean with gentle waves",
            seconds="5",
        )
        
        print(f"✅ 비디오 생성 완료!")
        print(f"   Video ID: {video.id}")
        print(f"   Status: {video.status}")
        
        # 에러 정보 출력
        if video.status == "failed":
            if hasattr(video, 'error') and video.error:
                error = video.error
                if isinstance(error, dict):
                    print(f"   ❌ 에러: {error.get('message', 'Unknown error')}")
                    if 'code' in error:
                        print(f"   에러 코드: {error['code']}")
                    if 'details' in error:
                        print(f"   상세 정보: {error['details']}")
                else:
                    print(f"   ❌ 에러: {error}")
            else:
                print(f"   ❌ 비디오 생성 실패 (에러 정보 없음)")
            
            # 디버그 정보 출력
            if hasattr(video, '_hidden_params') and video._hidden_params:
                hidden = video._hidden_params
                if 'debug_response' in hidden:
                    print(f"   디버그 응답: {hidden['debug_response']}")
        
        # 비용 정보 출력
        if hasattr(video, 'usage') and video.usage:
            usage = video.usage
            if isinstance(usage, dict) and "duration_seconds" in usage:
                duration = usage["duration_seconds"]
                print(f"   비디오 길이: {duration}초")
                # veo3-0 가격: $0.10/초
                estimated_cost = duration * 0.10
                print(f"   예상 비용: ${estimated_cost:.4f}")
    
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "Connection error" in error_msg:
            print(f"❌ 연결 오류: LiteLLM Proxy 서버가 실행되고 있지 않습니다.")
            print(f"   Proxy URL: {proxy_url}")
            print(f"   서버를 시작하려면: litellm --config config.yaml")
            sys.exit(1)
        else:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
