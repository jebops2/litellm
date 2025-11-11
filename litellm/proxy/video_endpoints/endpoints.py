#### Video Endpoints #####

import asyncio
import orjson
from fastapi import APIRouter, Depends, Request, Response, UploadFile, File
from fastapi.responses import ORJSONResponse
from typing import Optional, Dict, Any

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.image_endpoints.endpoints import batch_to_bytesio
from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
from litellm.proxy.common_utils.openai_endpoint_utils import (
    get_custom_llm_provider_from_request_body,
    get_custom_llm_provider_from_request_headers,
    get_custom_llm_provider_from_request_query,
)

router = APIRouter()


@router.post(
    "/vertexai_video",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos", "vertex_ai"],
)
async def vertexai_video_generation(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Vertex AI Veo video generation endpoint.
    
    This endpoint is specifically for Vertex AI Veo models:
    - veo-3.0-generate-001
    - veo-3.0-fast-generate-001
    - veo-3.0-generate-preview
    - veo-3.0-fast-generate-preview
    - veo-3.1-generate-preview
    - veo-3.1-fast-generate-preview
    
    This endpoint uses Proxy's authentication, logging, and budget control,
    but directly calls VertexVideoGeneration without going through
    the standard video_generation function to avoid affecting existing code.
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/vertexai_video" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "veo3-0",
            "prompt": "A beautiful sunset over the ocean"
        }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
    from litellm.secret_managers.main import get_secret_str
    from litellm.llms.vertex_ai.video_generation.video_generation_handler import (
        VertexVideoGeneration,
    )
    from litellm.types.videos.main import VideoObject
    import litellm

    # Read request body
    data = await _read_request_body(request=request)
    
    # Ensure model is set
    if "model" not in data:
        raise ValueError("model is required for /vertexai_video endpoint")
    
    # Force custom_llm_provider to vertex_ai for this endpoint
    data["custom_llm_provider"] = "vertex_ai"
    
    # Process request using ProxyBaseLLMRequestProcessing for authentication, logging, budget control
    processor = ProxyBaseLLMRequestProcessing(data=data)
    
    try:
        # Use base_process_llm_request's pre-call logic for auth, budget checks, logging
        data_with_logging, litellm_logging_obj = await processor.common_processing_pre_call_logic(
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            proxy_config=proxy_config,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            model=None,
            route_type="avideo_generation",
            version=version,
        )
        
        # During call hook
        await proxy_logging_obj.during_call_hook(
            data=data_with_logging,
            user_api_key_dict=user_api_key_dict,
            call_type="image_generation",  # Use image_generation as closest match for video_generation
        )
        
        # Extract parameters
        model_name_from_request = data_with_logging.get("model", "")  # This is model_name (e.g., "veo3-0")
        prompt = data_with_logging.get("prompt", "")
        if not prompt:
            raise ValueError("prompt is required for /vertexai_video endpoint")
        
        # Get model info from router to extract vertex_project and vertex_location
        # This works for all vertexai_video models (veo3-0, veo3-1, etc.)
        vertex_project = None
        vertex_location = None
        vertex_credentials = None
        actual_model = None  # This will be litellm_params.model (e.g., "vertex_ai/veo-3.0-generate-preview")
        
        if llm_router:
            try:
                # Method 1: Try to get deployment by model group name (e.g., "veo3-0")
                deployment = llm_router.get_deployment_by_model_group_name(model_name_from_request)
                
                if deployment is not None:
                    # Get from deployment object
                    litellm_params = deployment.litellm_params
                    actual_model = getattr(litellm_params, "model", None) or model_name_from_request
                    vertex_project = getattr(litellm_params, "vertex_project", None) or getattr(litellm_params, "vertex_ai_project", None)
                    vertex_location = getattr(litellm_params, "vertex_location", None) or getattr(litellm_params, "vertex_ai_location", None)
                    vertex_credentials = getattr(litellm_params, "vertex_credentials", None) or getattr(litellm_params, "vertex_ai_credentials", None)
                
                # Method 2: If not found, try to get from model list
                if not vertex_project or not vertex_location or not actual_model:
                    model_list = llm_router.get_model_list(model_name=model_name_from_request)
                    if model_list and len(model_list) > 0:
                        # Try all deployments in the list until we find vertex config
                        for deployment_item in model_list:
                            if isinstance(deployment_item, dict):
                                litellm_params = deployment_item.get("litellm_params", {})
                            elif hasattr(deployment_item, "litellm_params"):
                                litellm_params = deployment_item.litellm_params
                                # Convert to dict if it's an object
                                if hasattr(litellm_params, "model_dump"):
                                    litellm_params = litellm_params.model_dump()
                                elif hasattr(litellm_params, "dict"):
                                    litellm_params = litellm_params.dict()
                                else:
                                    litellm_params = {}
                            else:
                                continue
                            
                            # Extract vertex config from litellm_params
                            if isinstance(litellm_params, dict):
                                if not actual_model:
                                    actual_model = litellm_params.get("model") or model_name_from_request
                                if not vertex_project:
                                    vertex_project = litellm_params.get("vertex_project") or litellm_params.get("vertex_ai_project")
                                if not vertex_location:
                                    vertex_location = litellm_params.get("vertex_location") or litellm_params.get("vertex_ai_location")
                                if not vertex_credentials:
                                    vertex_credentials = litellm_params.get("vertex_credentials") or litellm_params.get("vertex_ai_credentials")
                            
                            # If we found all required config, break
                            if vertex_project and vertex_location and actual_model:
                                break
                
                # Method 3: Try to get by model ID if model is an ID
                if not vertex_project or not vertex_location or not actual_model:
                    try:
                        deployment_by_id = llm_router.get_deployment(model_id=model_name_from_request)
                        if deployment_by_id is not None:
                            litellm_params = deployment_by_id.litellm_params
                            if not actual_model:
                                actual_model = getattr(litellm_params, "model", None) or model_name_from_request
                            if not vertex_project:
                                vertex_project = getattr(litellm_params, "vertex_project", None) or getattr(litellm_params, "vertex_ai_project", None)
                            if not vertex_location:
                                vertex_location = getattr(litellm_params, "vertex_location", None) or getattr(litellm_params, "vertex_ai_location", None)
                            if not vertex_credentials:
                                vertex_credentials = getattr(litellm_params, "vertex_credentials", None) or getattr(litellm_params, "vertex_ai_credentials", None)
                    except Exception:
                        pass
                        
            except Exception:
                # Log the error but continue with fallback options
                pass
        
        # Fallback to environment variables
        if not vertex_project:
            vertex_project = get_secret_str("VERTEXAI_PROJECT")
        if not vertex_location:
            vertex_location = get_secret_str("VERTEXAI_LOCATION")
        if not vertex_credentials:
            vertex_credentials = get_secret_str("VERTEXAI_CREDENTIALS")
        
        # Fallback to global litellm settings
        if not vertex_project:
            vertex_project = litellm.vertex_project
        if not vertex_location:
            vertex_location = litellm.vertex_location
        
        # Fallback: if we couldn't find actual_model from deployment, use the request model
        if not actual_model:
            actual_model = model_name_from_request
        
        # Ensure actual_model is a valid string
        if not actual_model or not isinstance(actual_model, str):
            raise ValueError(f"Invalid model: {actual_model}. Model must be a non-empty string.")
        
        # Remove vertex_ai/ prefix from model name if present
        # litellm_params.model might be "vertex_ai/veo-3.0-generate-preview"
        # but Vertex AI API expects just "veo-3.0-generate-preview"
        if actual_model.startswith("vertex_ai/"):
            actual_model = actual_model.split("/", 1)[1]
        
        # Extract optional parameters
        optional_params = {}
        for key in ["seconds", "size", "user"]:
            if key in data_with_logging:
                optional_params[key] = data_with_logging[key]
        
        # Create model response
        model_response = VideoObject(
            id="",
            object="video",
            status="processing",
            created_at=0,
        )
        
        # Create handler and call video generation with Proxy's logging object
        vertex_video_generation = VertexVideoGeneration()
        
        result = await vertex_video_generation.video_generation(
            model=actual_model,  # Use actual_model (litellm_params.model) instead of model_name
            prompt=prompt,
            timeout=data_with_logging.get("timeout", 600),
            logging_obj=litellm_logging_obj,
            optional_params=optional_params,
            model_response=model_response,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            avideo_generation=True,
            api_base=None,  # Will be auto-generated from vertex_project and vertex_location
            client=None,
            extra_headers=None,
        )
        
        # Post-call processing
        hidden_params = getattr(result, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        
        if llm_router is not None:
            data_with_logging["deployment"] = llm_router.get_deployment(model_id=model_id)
        
        asyncio.create_task(
            proxy_logging_obj.update_request_status(
                litellm_call_id=data_with_logging.get("litellm_call_id", ""), status="success"
            )
        )
        
        # Post-call hook for logging usage, costs, etc.
        await proxy_logging_obj.post_call_success_hook(
            data=data_with_logging,
            response=result,
            user_api_key_dict=user_api_key_dict,
        )
        
        # Convert VideoObject to dict for JSON response
        if hasattr(result, "dict"):
            return result.dict()
        elif hasattr(result, "model_dump"):
            return result.model_dump()
        else:
            return result
        
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/videos",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.post(
    "/videos",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_generation(
    request: Request,
    fastapi_response: Response,
    input_reference: Optional[UploadFile] = File(None),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video generation endpoint for creating videos from text prompts.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Supports multiple providers including:
    - OpenAI (Sora models)
    - Vertex AI (Veo models: veo-2.0-generate-001, veo-3.0-generate-001, veo-3.0-fast-generate-001, veo-3.0-generate-preview, veo-3.0-fast-generate-preview, veo-3.1-generate-001, veo-3.1-fast-generate-001, veo-3.1-generate-preview, veo-3.1-fast-generate-preview)
    
    Example for Vertex AI Veo:
    ```bash
    curl -X POST "http://localhost:4000/v1/videos" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "vertex_ai/veo-3.0-generate-preview",
            "prompt": "A beautiful sunset over the ocean"
        }'
    ```
    
    Example for OpenAI:
    ```bash
    curl -X POST "http://localhost:4000/v1/videos" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "sora-2",
            "prompt": "A beautiful sunset over the ocean"
        }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body
    data = await _read_request_body(request=request)
    if input_reference is not None:
        input_reference_file = await batch_to_bytesio([input_reference])
        if input_reference_file:
            data["input_reference"] = input_reference_file[0]

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_generation",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/videos",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.get(
    "/videos",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_list(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video list endpoint for retrieving a list of videos.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/videos" \
        -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read query parameters
    query_params = dict(request.query_params)
    data: Dict[str, Any] = {"query_params": query_params}

    # Extract custom_llm_provider from headers, query params, or body
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or await get_custom_llm_provider_from_request_body(request=request)
    )
    if custom_llm_provider:
        data["custom_llm_provider"] = custom_llm_provider
    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_list",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/videos/{video_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.get(
    "/videos/{video_id}",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_status(
    video_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video status endpoint for retrieving video status and metadata.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/videos/video_123" \
        -H "Authorization: Bearer sk-1234"
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Create data with video_id
    data: Dict[str, Any] = {"video_id": video_id}

    # Extract custom_llm_provider from headers, query params, or body
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or await get_custom_llm_provider_from_request_body(request=request)
        or "openai"

    )
    if custom_llm_provider:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_status",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.get(
    "/v1/videos/{video_id}/content",
    dependencies=[Depends(user_api_key_auth)],
    response_class=Response,
    tags=["videos"],
)
@router.get(
    "/videos/{video_id}/content",
    dependencies=[Depends(user_api_key_auth)],
    response_class=Response,
    tags=["videos"],
)
async def video_content(
    video_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video content endpoint for downloading video content.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X GET "http://localhost:4000/v1/videos/video_123/content" \
        -H "Authorization: Bearer sk-1234" \
        --output video.mp4
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Create data with video_id
    data: Dict[str, Any] = {"video_id": video_id}

    # Extract custom_llm_provider from headers, query params, or body
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or await get_custom_llm_provider_from_request_body(request=request)
    )
    if custom_llm_provider:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        # Call the video content function directly to get raw bytes
        video_bytes = await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_content",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
        
        # Return raw video bytes with proper content type
        return Response(
            content=video_bytes,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=video_{video_id}.mp4"
            }
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )


@router.post(
    "/v1/videos/{video_id}/remix",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
@router.post(
    "/videos/{video_id}/remix",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["videos"],
)
async def video_remix(
    video_id: str,
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Video remix endpoint for remixing existing videos with new prompts.
    
    Follows the OpenAI Videos API spec:
    https://platform.openai.com/docs/api-reference/videos
    
    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/videos/video_123/remix" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "A new version with different colors"
        }'
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    # Read request body
    body = await request.body()
    data = orjson.loads(body)
    data["video_id"] = video_id

    # Extract custom_llm_provider from headers, query params, or body
    custom_llm_provider = (
        get_custom_llm_provider_from_request_headers(request=request)
        or get_custom_llm_provider_from_request_query(request=request)
        or data.get("custom_llm_provider")
    )
    if custom_llm_provider:
        data["custom_llm_provider"] = custom_llm_provider

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="avideo_remix",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
