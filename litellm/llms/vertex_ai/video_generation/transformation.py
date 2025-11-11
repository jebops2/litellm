from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import httpx

from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.router import GenericLiteLLMParams
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from ...base_llm.videos.transformation import BaseVideoConfig as _BaseVideoConfig
    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseVideoConfig = _BaseVideoConfig
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseVideoConfig = Any
    BaseLLMException = Any

import litellm
from ...base_llm.videos.transformation import BaseVideoConfig


class VertexAIVideoConfig(BaseVideoConfig):
    """
    Configuration class for Vertex AI Veo video generation.
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for Veo video generation.
        """
        return [
            "model",
            "prompt",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "vertex_project",
            "vertex_location",
            "vertex_credentials",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI parameters to Vertex AI parameters.
        """
        mapped_params = dict(video_create_optional_params)
        
        # Remove model and extra_headers as they're handled separately
        mapped_params.pop("model", None)
        mapped_params.pop("extra_headers", None)
        
        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and set up authentication headers.
        """
        # Vertex AI uses OAuth2 tokens, not API keys
        # Headers will be set by the handler
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Vertex AI video generation.
        This is handled by the handler, so we return api_base if provided.
        """
        if api_base:
            return api_base
        
        # Default will be set by handler
        return ""

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, List]:
        """
        Transform the video creation request for Vertex AI API.
        
        Vertex AI Veo uses:
        - POST to predictLongRunning endpoint
        - Request body: {"instances": [{"prompt": prompt}], "parameters": {...}}
        """
        # Remove model and extra_headers from optional params
        params = {
            k: v for k, v in video_create_optional_request_params.items()
            if k not in ["model", "extra_headers"]
        }
        
        # Transform snake_case to camelCase
        def snake_to_camel(snake_str: str) -> str:
            components = snake_str.split("_")
            return components[0] + "".join(word.capitalize() for word in components[1:])
        
        transformed_params = {}
        for key, value in params.items():
            if "_" in key:
                camel_case_key = snake_to_camel(key)
                transformed_params[camel_case_key] = value
            else:
                transformed_params[key] = value
        
        # Build request data
        request_data: Dict[str, Any] = {
            "instances": [{"prompt": prompt}],
        }
        
        if transformed_params:
            request_data["parameters"] = transformed_params
        
        # No files for Vertex AI video generation
        files_list: List[Tuple[str, Any]] = []
        
        return request_data, files_list

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the Vertex AI video creation response.
        
        Note: Vertex AI video generation uses long-running operations,
        so the initial response contains an operation name.
        The handler will poll the operation and return the final result.
        """
        response_data = raw_response.json()
        
        # Extract operation name
        operation_name = response_data.get("name", "")
        
        # Create a VideoObject with operation info
        video_obj_data = {
            "id": operation_name.split("/")[-1] if operation_name else "",
            "object": "video",
            "status": "processing",
            "created_at": int(__import__("time").time()),
            "model": model,
            "progress": 0,
            "_hidden_params": {
                "operation_name": operation_name,
            },
        }
        
        return VideoObject(**video_obj_data)

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video content request for Vertex AI API.
        
        Vertex AI Veo stores videos in GCS, so we need to retrieve the URI from the video object.
        """
        # For Vertex AI, video content is stored in GCS
        # The URI should be retrieved from the video object's _hidden_params
        url = f"{api_base.rstrip('/')}/{video_id}/content"
        params: Dict[str, Any] = {}
        return url, params

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the Vertex AI video content download response.
        Returns raw video content as bytes.
        """
        return raw_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """
        Transform the video remix request for Vertex AI API.
        
        Note: Vertex AI Veo may not support remix directly.
        """
        url = f"{api_base.rstrip('/')}/{video_id}/remix"
        data: Dict[str, Any] = {"prompt": prompt}
        if extra_body:
            data.update(extra_body)
        return url, data

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the Vertex AI video remix response.
        """
        response_data = raw_response.json()
        video_obj = VideoObject(**response_data)  # type: ignore[arg-type]
        
        # Create usage object with duration information
        usage_data = {}
        if hasattr(video_obj, 'seconds') and video_obj.seconds:
            try:
                usage_data["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage_data
        
        return video_obj

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """
        Transform the video list request for Vertex AI API.
        """
        url = api_base
        params: Dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = str(limit)
        if order is not None:
            params["order"] = order
        if extra_query:
            params.update(extra_query)
        return url, params

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> List[VideoObject]:
        """
        Transform the Vertex AI video list response.
        """
        response_data = raw_response.json()
        # OpenAI format: {"data": [VideoObject, ...], "has_more": bool}
        if isinstance(response_data, dict) and "data" in response_data:
            return [VideoObject(**item) for item in response_data["data"]]  # type: ignore[arg-type]
        # If it's a list directly
        if isinstance(response_data, list):
            return [VideoObject(**item) for item in response_data]  # type: ignore[arg-type]
        return []

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video delete request for Vertex AI API.
        
        Note: Vertex AI Veo may not support deletion directly.
        """
        url = f"{api_base.rstrip('/')}/{video_id}"
        data: Dict[str, Any] = {}
        return url, data

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the Vertex AI video delete response.
        """
        response_data = raw_response.json()
        video_obj = VideoObject(**response_data)  # type: ignore[arg-type]
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video retrieve request for Vertex AI API.
        """
        url = f"{api_base.rstrip('/')}/{video_id}"
        data: Dict[str, Any] = {}
        return url, data

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the Vertex AI video retrieve response.
        """
        response_data = raw_response.json()
        video_obj = VideoObject(**response_data)  # type: ignore[arg-type]
        
        # Create usage object with duration information
        usage_data = {}
        if hasattr(video_obj, 'seconds') and video_obj.seconds:
            try:
                usage_data["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage_data
        
        return video_obj

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )


def get_vertex_ai_video_config(model: Optional[str]) -> VertexAIVideoConfig:
    """
    Get Vertex AI video generation config for a given model.
    """
    return VertexAIVideoConfig()

