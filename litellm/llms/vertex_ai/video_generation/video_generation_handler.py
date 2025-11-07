import json
import time
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.videos.main import VideoObject


class VertexVideoGeneration(VertexLLM):
    """
    Handler for Vertex AI Veo video generation.
    
    Veo models use long-running operations, so we need to:
    1. Submit a request to predictLongRunning endpoint
    2. Poll the operation until it's complete
    3. Return the video object with the operation/result
    """
    
    def transform_optional_params(self, optional_params: Optional[dict]) -> dict:
        """
        Transform the optional params to the format expected by the Vertex AI API.
        For example, "aspect_ratio" is transformed to "aspectRatio".
        """
        if optional_params is None:
            return {}

        def snake_to_camel(snake_str: str) -> str:
            """Convert snake_case to camelCase"""
            components = snake_str.split("_")
            return components[0] + "".join(word.capitalize() for word in components[1:])

        transformed_params = {}
        for key, value in optional_params.items():
            if "_" in key:
                camel_case_key = snake_to_camel(key)
                transformed_params[camel_case_key] = value
            else:
                transformed_params[key] = value

        return transformed_params

    def _poll_operation_sync(
        self,
        operation_name: str,
        fetch_predict_operation_url: str,
        headers: Dict[str, str],
        timeout_secs: int,
        client: Optional[HTTPHandler] = None,
    ) -> Dict[str, Any]:
        """
        Poll Vertex AI operation until completion (sync).
        
        Uses fetchPredictOperation endpoint as per Google Cloud docs:
        https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation?hl=ko#response-body-poll-lro
        
        Args:
            operation_name: The full operation name (e.g., projects/.../operations/...)
            fetch_predict_operation_url: The fetchPredictOperation endpoint URL
            headers: Request headers (including auth)
            timeout_secs: Total timeout in seconds
            client: HTTP client to use
            
        Returns:
            Final operation response with completed video generation
        """
        if client is None:
            client = HTTPHandler(timeout=httpx.Timeout(timeout=timeout_secs, connect=5.0))
        
        start_time = time.time()
        poll_interval = 5  # Start with 5 seconds
        
        # Request body for fetchPredictOperation
        # According to Google Cloud docs, the field name is "operationName" not "name"
        request_body = {"operationName": operation_name}
        
        while time.time() - start_time < timeout_secs:
            # Poll the operation status using fetchPredictOperation (POST)
            response = client.post(
                url=fetch_predict_operation_url,
                headers=headers,
                data=json.dumps(request_body),
            )
            
            if response.status_code != 200:
                raise Exception(f"Error polling operation: {response.status_code} {response.text}")
            
            operation_data = response.json()
            
            # Check for errors
            if "error" in operation_data:
                error_msg = operation_data["error"].get("message", "Unknown error")
                raise Exception(f"Operation error: {error_msg}")
            
            # Check if operation is done
            is_done = operation_data.get("done", False)
            
            if is_done:
                return operation_data
            
            # Wait before next poll
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.2, 30)  # Cap at 30 seconds
        
        raise Exception(f"Operation timeout after {timeout_secs} seconds")

    async def _poll_operation_async(
        self,
        operation_name: str,
        fetch_predict_operation_url: str,
        headers: Dict[str, str],
        timeout_secs: int,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> Dict[str, Any]:
        """
        Poll Vertex AI operation until completion (async).
        
        Uses fetchPredictOperation endpoint as per Google Cloud docs:
        https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation?hl=ko#response-body-poll-lro
        
        Args:
            operation_name: The full operation name (e.g., projects/.../operations/...)
            fetch_predict_operation_url: The fetchPredictOperation endpoint URL
            headers: Request headers (including auth)
            timeout_secs: Total timeout in seconds
            client: HTTP client to use
            
        Returns:
            Final operation response with completed video generation
        """
        import asyncio
        
        if client is None:
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params={"timeout": httpx.Timeout(timeout=timeout_secs, connect=5.0)},
            )
        
        start_time = time.time()
        poll_interval = 5  # Start with 5 seconds
        
        # Request body for fetchPredictOperation
        # According to Google Cloud docs, the field name is "operationName" not "name"
        request_body = {"operationName": operation_name}
        
        while time.time() - start_time < timeout_secs:
            # Poll the operation status using fetchPredictOperation (POST)
            response = await client.post(
                url=fetch_predict_operation_url,
                headers=headers,
                data=json.dumps(request_body),
            )
            
            if response.status_code != 200:
                raise Exception(f"Error polling operation: {response.status_code} {response.text}")
            
            operation_data = response.json()
            
            # Check for errors
            if "error" in operation_data:
                error_msg = operation_data["error"].get("message", "Unknown error")
                raise Exception(f"Operation error: {error_msg}")
            
            # Check if operation is done
            is_done = operation_data.get("done", False)
            
            if is_done:
                return operation_data
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.2, 30)  # Cap at 30 seconds
        
        raise Exception(f"Operation timeout after {timeout_secs} seconds")

    def _process_video_generation_response(
        self,
        operation_response: Dict[str, Any],
        model: str,
        operation_name: Optional[str] = None,
    ) -> VideoObject:
        """
        Process the video generation operation response into a VideoObject.
        
        Args:
            operation_response: The operation response from Vertex AI
            model: The model name used
            operation_name: The operation name (if available)
            
        Returns:
            VideoObject with video generation details
        """
        import time as time_module
        
        # Extract operation name if not provided
        if operation_name is None:
            operation_name = operation_response.get("name", "")
        
        # Check if operation is done
        is_done = operation_response.get("done", False)
        
        # Check for operation errors
        if "error" in operation_response:
            error_info = operation_response.get("error", {})
            error_msg = error_info.get("message", "Unknown error")
            error_code = error_info.get("code", 0)
            raise Exception(f"Operation failed with error {error_code}: {error_msg}")
        
        # Create base video object
        video_obj_data: Dict[str, Any] = {
            "id": operation_name.split("/")[-1] if operation_name else "",
            "object": "video",
            "model": model,
            "created_at": int(time_module.time()),
        }
        
        if is_done:
            # Extract video URI from response
            # According to Google Cloud docs, the response format is:
            # {
            #   "name": "...",
            #   "done": true,
            #   "response": {
            #     "@type": "type.googleapis.com/cloud.ai.large_models.vision.GenerateVideoResponse",
            #     "raiMediaFilteredCount": 0,
            #     "videos": [
            #       {
            #         "gcsUri": "...",
            #         "mimeType": "video/mp4"
            #       }
            #     ]
            #   }
            # }
            try:
                response_data = operation_response.get("response", {})
                
                # Try multiple response formats for compatibility
                videos = None
                if "videos" in response_data:
                    # Standard format from docs
                    videos = response_data.get("videos", [])
                elif "generateVideoResponse" in response_data:
                    # Alternative format
                    generate_response = response_data.get("generateVideoResponse", {})
                    if "generatedSamples" in generate_response:
                        generated_samples = generate_response.get("generatedSamples", [])
                        if generated_samples:
                            video_data = generated_samples[0].get("video", {})
                            videos = [video_data] if video_data else []
                    elif "videos" in generate_response:
                        videos = generate_response.get("videos", [])
                
                # Check for filtered videos
                rai_media_filtered_count = response_data.get("raiMediaFilteredCount", 0)
                
                if videos and len(videos) > 0:
                    # Extract video URI (can be gcsUri or uri)
                    video_item = videos[0]
                    video_uri = video_item.get("gcsUri") or video_item.get("uri") or ""
                    
                    if video_uri:
                        # Store video URI in hidden params for later retrieval
                        video_obj_data["_hidden_params"] = {
                            "video_uri": video_uri,
                            "operation_name": operation_name,
                            "mime_type": video_item.get("mimeType", "video/mp4"),
                        }
                        
                        # Set status to completed
                        video_obj_data["status"] = "completed"
                        video_obj_data["completed_at"] = int(time_module.time())
                        video_obj_data["progress"] = 100
                    else:
                        video_obj_data["status"] = "failed"
                        video_obj_data["error"] = {"message": "Video URI not found in response"}
                else:
                    # No videos in response - could be filtered or error
                    error_msg = "No video samples in response"
                    if rai_media_filtered_count > 0:
                        error_msg += f" (filtered: {rai_media_filtered_count})"
                    video_obj_data["status"] = "failed"
                    video_obj_data["error"] = {"message": error_msg}
                    # Log the full response for debugging
                    video_obj_data["_hidden_params"] = {
                        "operation_name": operation_name,
                        "debug_response": operation_response,
                    }
            except (KeyError, IndexError, TypeError) as e:
                video_obj_data["status"] = "failed"
                video_obj_data["error"] = {"message": f"Error parsing response: {str(e)}"}
                # Log the full response for debugging
                video_obj_data["_hidden_params"] = {
                    "operation_name": operation_name,
                    "debug_response": operation_response,
                    "parse_error": str(e),
                }
        else:
            # Operation still in progress
            video_obj_data["status"] = "processing"
            video_obj_data["progress"] = 0
        
        # Create usage object with duration information for cost calculation
        usage_data = {}
        if "seconds" in video_obj_data:
            try:
                usage_data["duration_seconds"] = float(video_obj_data["seconds"])
            except (ValueError, TypeError):
                pass
        
        video_obj_data["usage"] = usage_data
        
        return VideoObject(**video_obj_data)

    def video_generation(
        self,
        prompt: str,
        api_base: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        model_response: Any,  # VideoObject
        logging_obj: Any,
        model: str,
        client: Optional[Any] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        avideo_generation=False,
        extra_headers: Optional[dict] = None,
    ) -> VideoObject:
        """
        Generate video using Vertex AI Veo models (sync).
        
        Veo models use long-running operations:
        1. Submit request to predictLongRunning endpoint
        2. Poll operation until complete
        3. Return video object
        """
        if avideo_generation is True:
            return self.avideo_generation(  # type: ignore
                prompt=prompt,
                api_base=api_base,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_credentials=vertex_credentials,
                model=model,
                client=client,
                optional_params=optional_params,
                timeout=timeout,
                logging_obj=logging_obj,
                model_response=model_response,
                extra_headers=extra_headers,
            )

        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            sync_handler: HTTPHandler = HTTPHandler(**_params)  # type: ignore
        else:
            sync_handler = client  # type: ignore

        # Get auth and URL
        auth_header: Optional[str] = None
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=None,
            auth_header=auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider="vertex_ai",
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode="video_generation",
        )
        
        # Transform optional params to camelCase format
        optional_params = self.transform_optional_params(optional_params)

        # Build request data
        request_data: Dict[str, Any] = {
            "instances": [{"prompt": prompt}],
        }
        
        if optional_params:
            request_data["parameters"] = optional_params

        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": optional_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        # Submit video generation request
        response = sync_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        json_response = response.json()
        
        # Extract operation name
        operation_name = json_response.get("name", "")
        if not operation_name:
            raise Exception(f"No operation name in response: {json_response}")
        
        # Build fetchPredictOperation URL
        # According to Google Cloud docs: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation?hl=ko#response-body-poll-lro
        # Operation name format: projects/{project}/locations/{location}/publishers/google/models/{model}/operations/{operation_id}
        # fetchPredictOperation endpoint: https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:fetchPredictOperation
        # We need to extract project, location, and model from operation_name
        operation_parts = operation_name.split("/")
        project_id = None
        location_id = None
        model_id = None
        
        if "projects" in operation_parts:
            project_idx = operation_parts.index("projects")
            if project_idx + 1 < len(operation_parts):
                project_id = operation_parts[project_idx + 1]
        
        if "locations" in operation_parts:
            location_idx = operation_parts.index("locations")
            if location_idx + 1 < len(operation_parts):
                location_id = operation_parts[location_idx + 1]
        
        if "models" in operation_parts:
            models_idx = operation_parts.index("models")
            if models_idx + 1 < len(operation_parts):
                model_id = operation_parts[models_idx + 1]
        
        # Fallback to vertex_project, vertex_location, and model if not found in operation_name
        if not project_id:
            project_id = vertex_project
        if not location_id:
            location_id = vertex_location
        if not model_id:
            model_id = model
        
        if not project_id or not location_id or not model_id:
            raise Exception(f"Could not extract project, location, or model from operation_name: {operation_name}")
        
        # Build fetchPredictOperation endpoint URL
        fetch_predict_operation_url = f"https://{location_id}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location_id}/publishers/google/models/{model_id}:fetchPredictOperation"
        
        # Poll operation until complete
        timeout_secs = timeout if timeout else 600
        operation_response = self._poll_operation_sync(
            operation_name=operation_name,
            fetch_predict_operation_url=fetch_predict_operation_url,
            headers=headers,
            timeout_secs=timeout_secs,
            client=sync_handler,
        )
        
        # Process response into VideoObject
        return self._process_video_generation_response(
            operation_response=operation_response,
            model=model,
            operation_name=operation_name,
        )

    async def avideo_generation(
        self,
        prompt: str,
        api_base: Optional[str],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        model_response: Any,  # VideoObject
        logging_obj: Any,
        model: str,
        client: Optional[AsyncHTTPHandler] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[int] = None,
        extra_headers: Optional[dict] = None,
    ) -> VideoObject:
        """
        Generate video using Vertex AI Veo models (async).
        
        Veo models use long-running operations:
        1. Submit request to predictLongRunning endpoint
        2. Poll operation until complete
        3. Return video object
        """
        if client is None:
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    _httpx_timeout = httpx.Timeout(timeout)
                    _params["timeout"] = _httpx_timeout
            else:
                _params["timeout"] = httpx.Timeout(timeout=600.0, connect=5.0)

            self.async_handler = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.VERTEX_AI,
                params=_params,
            )
        else:
            self.async_handler = client  # type: ignore

        # Get auth and URL
        auth_header: Optional[str] = None
        auth_header, _ = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai",
        )
        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=None,
            auth_header=auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=False,
            custom_llm_provider="vertex_ai",
            api_base=api_base,
            should_use_v1beta1_features=False,
            mode="video_generation",
        )

        # Transform optional params to camelCase format
        optional_params = self.transform_optional_params(optional_params)

        # Build request data
        request_data: Dict[str, Any] = {
            "instances": [{"prompt": prompt}],
        }
        
        if optional_params:
            request_data["parameters"] = optional_params

        headers = self.set_headers(auth_header=auth_header, extra_headers=extra_headers)

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": optional_params,
                "api_base": api_base,
                "headers": headers,
            },
        )

        # Submit video generation request
        response = await self.async_handler.post(
            url=api_base,
            headers=headers,
            data=json.dumps(request_data),
        )

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        json_response = response.json()
        
        # Extract operation name
        operation_name = json_response.get("name", "")
        if not operation_name:
            raise Exception(f"No operation name in response: {json_response}")
        
        # Build fetchPredictOperation URL
        # According to Google Cloud docs: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation?hl=ko#response-body-poll-lro
        # Operation name format: projects/{project}/locations/{location}/publishers/google/models/{model}/operations/{operation_id}
        # fetchPredictOperation endpoint: https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:fetchPredictOperation
        # We need to extract project, location, and model from operation_name
        operation_parts = operation_name.split("/")
        project_id = None
        location_id = None
        model_id = None
        
        if "projects" in operation_parts:
            project_idx = operation_parts.index("projects")
            if project_idx + 1 < len(operation_parts):
                project_id = operation_parts[project_idx + 1]
        
        if "locations" in operation_parts:
            location_idx = operation_parts.index("locations")
            if location_idx + 1 < len(operation_parts):
                location_id = operation_parts[location_idx + 1]
        
        if "models" in operation_parts:
            models_idx = operation_parts.index("models")
            if models_idx + 1 < len(operation_parts):
                model_id = operation_parts[models_idx + 1]
        
        # Fallback to vertex_project, vertex_location, and model if not found in operation_name
        if not project_id:
            project_id = vertex_project
        if not location_id:
            location_id = vertex_location
        if not model_id:
            model_id = model
        
        if not project_id or not location_id or not model_id:
            raise Exception(f"Could not extract project, location, or model from operation_name: {operation_name}")
        
        # Build fetchPredictOperation endpoint URL
        fetch_predict_operation_url = f"https://{location_id}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location_id}/publishers/google/models/{model_id}:fetchPredictOperation"
        
        # Poll operation until complete
        timeout_secs = timeout if timeout else 600
        operation_response = await self._poll_operation_async(
            operation_name=operation_name,
            fetch_predict_operation_url=fetch_predict_operation_url,
            headers=headers,
            timeout_secs=timeout_secs,
            client=self.async_handler,
        )
        
        # Process response into VideoObject
        return self._process_video_generation_response(
            operation_response=operation_response,
            model=model,
            operation_name=operation_name,
        )

