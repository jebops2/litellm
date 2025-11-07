#!/usr/bin/env python3
import os
import requests

def test_veo_video_generation_custom_endpoint():
    """Custom /vertexai_video endpoint 사용 예시"""
    proxy_url = "http://localhost:4000"
    api_key = "sk-TGGGa9SPl-aW8xJY0m9Ifg"
    model = "veo3-0"
    prompt = "A cat playing with a ball of yarn in a sunny garden"
    
    # Custom endpoint에 직접 HTTP 요청
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "seconds": 8,
        "size": "720x1280"
    }
    
    try:
        response = requests.post(
            f"{proxy_url}/vertexai_video",
            headers=headers,
            json=payload,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()
        
        # 기본 정보 출력
        print(f"Video ID: {result.get('id', 'N/A')}")
        print(f"Status: {result.get('status', 'N/A')}")
        print(f"Model: {result.get('model', 'N/A')}")
        print(f"Created at: {result.get('created_at', 'N/A')}")
        
        # Progress 정보 출력
        if "progress" in result:
            print(f"Progress: {result['progress']}%")
        
        # Video URI 확인 (_hidden_params 안에 있을 수 있음)
        video_uri = None
        if "video_uri" in result:
            video_uri = result["video_uri"]
        elif "_hidden_params" in result and isinstance(result["_hidden_params"], dict):
            video_uri = result["_hidden_params"].get("video_uri")
        
        if video_uri:
            print(f"Video URI: {video_uri}")
        else:
            print("Video URI: Not available (may be available after processing completes)")
        
        # 에러 정보 확인
        if "error" in result and result["error"]:
            print(f"Error: {result['error']}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {error_detail}")
            except Exception:
                print(f"Error response: {e.response.text}")
        raise


if __name__ == "__main__":
    test_veo_video_generation_custom_endpoint()
