use crate::auth::middleware::{AuthMiddleware, AuthenticatedUser};
use crate::cache::cache_service::CacheService;
use crate::pyprocess::model::Model;
use crate::storage::s3_service::S3Service;
use actix_files::Files;
use actix_multipart::Multipart;
use actix_web::{http::header, web, Error, HttpResponse};
use futures::{stream, StreamExt, TryStreamExt};
use log::{error, info, warn};
use serde::Serialize;
use serde_json::{json, Value};
use shared::{InferenceResponse, ProcessingMode};
use std::io::Write;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Debug, Clone)]
struct InferenceContext {
    user_id: Option<Uuid>,
    use_cache: bool,
}

impl InferenceContext {
    fn authenticated(user_id: Uuid) -> Self {
        Self {
            user_id: Some(user_id),
            use_cache: true,
        }
    }

    fn public() -> Self {
        Self {
            user_id: None,
            use_cache: false,
        }
    }
}

pub fn configure_routes(
    cfg: &mut web::ServiceConfig,
    frontend_dir: String,
    auth_middleware: AuthMiddleware,
) {
    cfg
        // Favicon route - handle favicon requests explicitly
        .route("/favicon.ico", web::get().to(handle_favicon))
        // Public API endpoints (no authentication required) - now with streaming
        .service(
            web::scope("/public")
                .route("/inference", web::post().to(handle_inference_public_stream)),
        )
        // Protected API endpoints (authentication required) - unified streaming
        .service(
            web::scope("/api")
                .wrap(auth_middleware.clone())
                .route(
                    "/inference",
                    web::post().to(handle_inference_authenticated_stream),
                )
                .route("/cache/history", web::get().to(handle_cache_history))
                .route(
                    "/cache/image/{image_hash}",
                    web::get().to(handle_cached_image),
                )
                .route(
                    "/cache/image/{image_hash}",
                    web::delete().to(handle_delete_cached_image),
                )
                .route("/cache/clear", web::delete().to(handle_clear_user_cache)),
        )
        // Authentication routes - mixed public and protected
        .service(
            web::scope("/auth")
                // Public Cognito routes (no middleware)
                .route(
                    "/login",
                    web::get().to(crate::auth::cognito_routes::cognito_login_redirect),
                )
                .route(
                    "/cognito/callback",
                    web::get().to(crate::auth::cognito_routes::cognito_auth_callback),
                )
                // Protected endpoints (with middleware applied individually)
                .route(
                    "/me",
                    web::get()
                        .to(crate::auth::routes::me)
                        .wrap(auth_middleware.clone()),
                )
                .route(
                    "/logout",
                    web::post()
                        .to(crate::auth::cognito_routes::cognito_logout)
                        .wrap(auth_middleware.clone()),
                )
                .route(
                    "/refresh",
                    web::post()
                        .to(crate::auth::cognito_routes::cognito_refresh_token)
                        .wrap(auth_middleware),
                ),
        )
        // Serve the SPA - serve all static assets and index.html from the frontend dist directory
        .service(
            Files::new("/", frontend_dir)
                .index_file("index.html")
                .default_handler(web::route().to(spa_handler)),
        );
}

async fn handle_favicon() -> Result<actix_files::NamedFile, actix_web::Error> {
    let frontend_dir = if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        format!("{}/../frontend/dist", manifest_dir)
    } else {
        "/usr/src/app/frontend/dist".to_string()
    };

    let favicon_path = format!("{}/favicon.ico", frontend_dir);

    if std::path::Path::new(&favicon_path).exists() {
        match actix_files::NamedFile::open(&favicon_path) {
            Ok(file) => return Ok(file),
            Err(e) => {
                warn!("Failed to serve favicon.ico: {:?}", e);
            }
        }
    }

    warn!("Favicon not found at {}", favicon_path);
    Err(actix_web::error::ErrorNotFound("Favicon not found"))
}

// Handler for SPA routing - serves index.html for all unmatched routes
async fn spa_handler() -> Result<actix_files::NamedFile, actix_web::Error> {
    let frontend_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let frontend_dir = format!("{}/../frontend/dist", frontend_dir);
    let index_path = format!("{}/index.html", frontend_dir);

    actix_files::NamedFile::open(index_path).map_err(|e| {
        actix_web::error::ErrorInternalServerError(format!("Failed to serve index.html: {}", e))
    })
}

async fn parse_multipart_payload(
    mut payload: Multipart,
) -> Result<(Vec<(Vec<u8>, String, String, u64)>, ProcessingMode), Error> {
    let mut images_with_metadata = Vec::new();
    let mut processing_mode = ProcessingMode::IntermediateFusionEnsemble;
    let mut file_counter = 1u64;

    while let Ok(Some(mut field)) = payload.try_next().await {
        let field_start = std::time::Instant::now();
        let content_disposition = field.content_disposition();
        let field_name = content_disposition
            .as_ref()
            .and_then(|cd| cd.get_name())
            .unwrap_or("image");

        match field_name {
            "mode" => {
                let mut mode_data = Vec::new();
                while let Some(chunk) = field.next().await {
                    mode_data.write_all(&chunk?)?;
                }
                if let Ok(mode_str) = String::from_utf8(mode_data) {
                    processing_mode = match mode_str.trim() {
                        "late_fusion" => ProcessingMode::LateFusionEnsemble,
                        _ => ProcessingMode::IntermediateFusionEnsemble,
                    };
                    info!("Processing mode set to: {:?}", processing_mode);
                }
            }
            "image" => {
                let filename = content_disposition
                    .as_ref()
                    .and_then(|cd| cd.get_filename())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "uploaded_image".to_string());

                let mut image_data = Vec::new();
                let mut chunk_count = 0;
                while let Some(chunk) = field.next().await {
                    image_data.write_all(&chunk?)?;
                    chunk_count += 1;
                }

                if !image_data.is_empty() {
                    let mime_type = if filename.to_lowercase().ends_with(".png") {
                        "image/png"
                    } else if filename.to_lowercase().ends_with(".webp") {
                        "image/webp"
                    } else if filename.to_lowercase().ends_with(".gif") {
                        "image/gif"
                    } else {
                        "image/jpeg"
                    }
                    .to_string();

                    images_with_metadata.push((image_data, filename, mime_type, file_counter));
                    file_counter += 1;
                }
            }
            _ => {
                info!("Ignoring unknown multipart field: {}", field_name);
            }
        }
    }

    Ok((images_with_metadata, processing_mode))
}

async fn handle_inference_streaming(
    model: web::Data<Arc<Mutex<Model>>>,
    payload: Multipart,
    cache_service: Option<web::Data<CacheService>>,
    context: InferenceContext,
) -> Result<HttpResponse, Error> {
    let request_start = std::time::Instant::now();
    let context_desc = if context.use_cache {
        format!("authenticated user: {:?}", context.user_id)
    } else {
        "public (no caching)".to_string()
    };
    info!(
        "Processing streaming inference request for {} at {:?}",
        context_desc, request_start
    );

    let (images_with_metadata, processing_mode) = parse_multipart_payload(payload).await?;

    if images_with_metadata.is_empty() {
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "No image data received.".to_string(),
        }));
    }

    info!(
        "Processing {} images concurrently with streaming mode: {:?} (multipart parsing took {:?}ms)",
        images_with_metadata.len(),
        processing_mode,
        request_start.elapsed().as_millis()
    );

    let (tx, rx) = mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let spawn_start = std::time::Instant::now();
        info!("Spawning concurrent processing tasks at {:?}", spawn_start);

        let mut task_handles = Vec::new();

        for (image_data, filename, mime_type, file_id) in images_with_metadata {
            let filename_clone = filename.clone();

            let model_clone = model.clone();
            let cache_service_clone = cache_service.clone();
            let context_clone = context.clone();
            let processing_mode_clone = processing_mode.clone();
            let tx_clone = tx.clone();
            let spawn_start_clone = spawn_start;

            // Spawn each task as a separate tokio task for concurrency
            let handle = tokio::spawn(async move {
                let task_id = Uuid::new_v4();
                let image_hash = S3Service::calculate_image_hash(&image_data);

                let result_json = if context_clone.use_cache && cache_service_clone.is_some() {
                    let cache_service = cache_service_clone.as_ref().unwrap();
                    let user_id = context_clone.user_id.unwrap();

                    match cache_service
                        .get_cached_inference(user_id, &image_hash, &processing_mode_clone)
                        .await
                    {
                        Ok(cached_response) => {
                            info!("Cache hit for file_id: {}", file_id);
                            json!({
                                "file_id": file_id,
                                "task": { "id": task_id.to_string(), "status": "completed" },
                                "inference": cached_response,
                                "cached": true
                            })
                        }
                        Err(_) => {
                            if let Err(e) = cache_service
                                .cache_image(
                                    user_id,
                                    &image_data,
                                    filename.clone(),
                                    mime_type.clone(),
                                )
                                .await
                            {
                                warn!("Failed to cache image '{}': {:?}", filename, e);
                            }
                            let params = ProcessImageParams::new(
                                &model_clone,
                                &image_data,
                                &filename,
                                &image_hash,
                                &processing_mode_clone,
                                file_id,
                                task_id,
                                &context_clone,
                                cache_service_clone.as_ref(),
                            );
                            process_image_inference(params).await
                        }
                    }
                } else {
                    let params = ProcessImageParams::new(
                        &model_clone,
                        &image_data,
                        &filename,
                        &image_hash,
                        &processing_mode_clone,
                        file_id,
                        task_id,
                        &context_clone,
                        None,
                    );
                    process_image_inference(params).await
                };

                // Send result immediately when this task completes
                let sse_data = format!("data: {}\n\n", result_json);
                if let Err(e) = tx_clone.send(sse_data) {
                    error!("Failed to send SSE data for file_id {}: {:?}", file_id, e);
                } else {
                    info!(
                        "Sent result for file_id: {} at {:?}ms from spawn",
                        file_id,
                        spawn_start_clone.elapsed().as_millis()
                    );
                }
            });

            task_handles.push(handle);

            info!(
                "Spawned task for image '{}' (file_id: {}) at {:?}ms from spawn",
                filename_clone,
                file_id,
                spawn_start.elapsed().as_millis()
            );
        }

        for handle in task_handles {
            if let Err(e) = handle.await {
                error!("Task failed: {:?}", e);
            }
        }

        if let Err(e) = tx.send("data: {\"type\": \"complete\"}\n\n".to_string()) {
            error!("Failed to send completion signal: {:?}", e);
        } else {
            info!("Sent completion signal after all concurrent tasks finished.");
        }
    });

    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|data| (Ok::<_, Error>(web::Bytes::from(data)), rx))
    });

    Ok(HttpResponse::Ok()
        .insert_header((header::CONTENT_TYPE, "text/event-stream"))
        .insert_header((header::CACHE_CONTROL, "no-cache"))
        .insert_header((header::CONNECTION, "keep-alive"))
        .streaming(stream))
}

struct ProcessImageParams<'a> {
    model: &'a web::Data<Arc<Mutex<Model>>>,
    image_data: &'a [u8],
    filename: &'a str,
    image_hash: &'a str,
    processing_mode: &'a ProcessingMode,
    file_id: u64,
    task_id: Uuid,
    context: &'a InferenceContext,
    cache_service: Option<&'a web::Data<CacheService>>,
}

impl<'a> ProcessImageParams<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: &'a web::Data<Arc<Mutex<Model>>>,
        image_data: &'a [u8],
        filename: &'a str,
        image_hash: &'a str,
        processing_mode: &'a ProcessingMode,
        file_id: u64,
        task_id: Uuid,
        context: &'a InferenceContext,
        cache_service: Option<&'a web::Data<CacheService>>,
    ) -> Self {
        Self {
            model,
            image_data,
            filename,
            image_hash,
            processing_mode,
            file_id,
            task_id,
            context,
            cache_service,
        }
    }
}

async fn process_image_inference(params: ProcessImageParams<'_>) -> Value {
    let start_time = std::time::Instant::now();
    info!(
        "Attempting to acquire model lock for file_id: {} at {:?}ms",
        params.file_id,
        start_time.elapsed().as_millis()
    );

    // Clone the model Arc for use in spawn_blocking
    let model_clone = params.model.clone();
    let image_data = params.image_data.to_vec();
    let processing_mode = params.processing_mode.clone();
    let file_id = params.file_id;
    let image_hash = params.image_hash.to_string();
    let use_cache = params.context.use_cache;
    let filename = params.filename.to_string();
    let task_id = params.task_id;

    info!(
        "Starting model inference for file_id: {} at {:?}ms",
        file_id,
        start_time.elapsed().as_millis()
    );

    // Use spawn_blocking for inference to prevent blocking the async runtime
    let inference_result = tokio::task::spawn_blocking(move || {
        let mut model_guard = match model_clone.lock() {
            Ok(guard) => guard,
            Err(e) => {
                return Err(format!("Failed to lock model: {}", e));
            }
        };

        if let Err(e) = model_guard.persistent_load(&processing_mode) {
            return Err(format!("Failed to load models: {:?}", e));
        }

        match model_guard.inference(&image_data, &processing_mode) {
            Ok(predictions) => {
                let (is_ai, confidence) = model_guard.calculate_result(&predictions);
                let inference_response = InferenceResponse {
                    predictions: predictions.clone(),
                    class_labels: vec!["AI Generated".into(), "Human Created".into()],
                    is_ai,
                    confidence,
                    image_hash: if use_cache { Some(image_hash) } else { None },
                };

                Ok((predictions, is_ai, confidence, inference_response))
            }
            Err(e) => Err(format!("Inference error: {:?}", e)),
        }
    })
    .await;

    let inference_result = match inference_result {
        Ok(Ok(result)) => {
            info!(
                "Completed model inference for file_id: {} at {:?}ms",
                file_id,
                start_time.elapsed().as_millis()
            );
            result
        }
        Ok(Err(e)) => {
            error!(
                "Model inference error for streaming image '{}': {}",
                filename, e
            );
            return json!({
                "file_id": file_id,
                "task": { "id": task_id.to_string(), "status": "error" },
                "error": e
            });
        }
        Err(e) => {
            error!(
                "Spawn blocking error for streaming image '{}': {:?}",
                filename, e
            );
            return json!({
                "file_id": file_id,
                "task": { "id": task_id.to_string(), "status": "error" },
                "error": format!("Task error: {:?}", e)
            });
        }
    };

    // Now handle the result and cache it (outside the mutex scope)
    let (predictions, is_ai, confidence, inference_response) = inference_result;

    // Cache the result
    if params.context.use_cache && params.cache_service.is_some() {
        let cache_service = params.cache_service.unwrap();
        let user_id = params.context.user_id.unwrap();

        if let Err(e) = cache_service
            .cache_inference_result(
                user_id,
                params.image_hash.to_string(),
                params.processing_mode,
                &predictions,
                is_ai,
                confidence,
            )
            .await
        {
            warn!("Failed to cache streaming inference result: {:?}", e);
        }
    }

    json!({
        "file_id": params.file_id,
        "task": { "id": params.task_id.to_string(), "status": "completed" },
        "inference": inference_response,
        "cached": false
    })
}

async fn handle_inference_authenticated_stream(
    model: web::Data<Arc<Mutex<Model>>>,
    payload: Multipart,
    cache_service: web::Data<CacheService>,
    user: AuthenticatedUser,
) -> Result<HttpResponse, Error> {
    let context = InferenceContext::authenticated(user.0);
    handle_inference_streaming(model, payload, Some(cache_service), context).await
}

async fn handle_inference_public_stream(
    model: web::Data<Arc<Mutex<Model>>>,
    payload: Multipart,
) -> Result<HttpResponse, Error> {
    let context = InferenceContext::public();
    handle_inference_streaming(model, payload, None, context).await
}

async fn handle_cache_history(
    user: AuthenticatedUser,
    cache_service: web::Data<CacheService>,
) -> Result<HttpResponse, Error> {
    let user_id = user.0;
    info!("Fetching cache history for user: {}", user_id);

    match cache_service.get_user_cache_history(user_id).await {
        Ok(cache_entries) => {
            info!(
                "Found {} cached entries for user: {}",
                cache_entries.len(),
                user_id
            );

            // Transform cache entries into frontend-compatible format
            let mut session_data = json!({
                "images": {},
                "results": {}
            });

            let mut file_id_counter = 1u64;

            for entry in cache_entries {
                // Create a file entry compatible with frontend FileData structure
                let file_entry = json!({
                    "id": file_id_counter,
                    "name": entry.file_name,
                    "size": entry.file_size,
                    "mime_type": entry.mime_type,
                    "image_hash": entry.image_hash,
                    "s3_key": entry.s3_key,
                    "s3_bucket": entry.s3_bucket,
                    "created_at": entry.image_created_at,
                    "preview_url": format!("/api/cache/image/{}", entry.image_hash)
                });

                session_data["images"][file_id_counter.to_string()] = file_entry;

                // Add inference results if available
                if !entry.inference_results.is_empty() {
                    // Use the most recent inference result (they're sorted by created_at)
                    if let Some(latest_result) = entry.inference_results.first() {
                        let predictions: Vec<f32> =
                            serde_json::from_value(latest_result.predictions.clone())
                                .unwrap_or_default();

                        let inference_response = json!({
                            "predictions": predictions,
                            "class_labels": ["AI Generated", "Human Created"],
                            "is_ai": latest_result.is_ai,
                            "confidence": latest_result.confidence,
                            "image_hash": entry.image_hash
                        });

                        session_data["results"][file_id_counter.to_string()] = inference_response;
                    }
                }

                file_id_counter += 1;
            }

            Ok(HttpResponse::Ok().json(session_data))
        }
        Err(e) => {
            error!(
                "Failed to fetch cache history for user {}: {:?}",
                user_id, e
            );
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to fetch cache history".to_string(),
            }))
        }
    }
}

async fn handle_cached_image(
    path: web::Path<String>,
    user: AuthenticatedUser,
    cache_service: web::Data<CacheService>,
) -> Result<HttpResponse, Error> {
    let image_hash = path.into_inner();
    let user_id = user.0;

    info!("Serving cached image {} for user: {}", image_hash, user_id);

    // Get image metadata from cache
    match cache_service
        .get_cached_image_metadata(user_id, &image_hash)
        .await
    {
        Ok(image_entry) => {
            // Get image data from S3
            match cache_service
                .get_cached_image_data(&image_entry.s3_key)
                .await
            {
                Ok(image_data) => {
                    info!("Successfully retrieved cached image: {}", image_hash);
                    Ok(HttpResponse::Ok()
                        .content_type(&*image_entry.mime_type)
                        .body(image_data))
                }
                Err(e) => {
                    error!("Failed to retrieve image data from S3: {:?}", e);
                    Ok(HttpResponse::NotFound().json(ErrorResponse {
                        error: "Image data not found".to_string(),
                    }))
                }
            }
        }
        Err(e) => {
            error!("Failed to get cached image metadata: {:?}", e);
            Ok(HttpResponse::NotFound().json(ErrorResponse {
                error: "Image not found".to_string(),
            }))
        }
    }
}

async fn handle_delete_cached_image(
    path: web::Path<String>,
    user: AuthenticatedUser,
    cache_service: web::Data<CacheService>,
) -> Result<HttpResponse, Error> {
    let image_hash = path.into_inner();
    let user_id = user.0;

    info!("Deleting cached image {} for user: {}", image_hash, user_id);

    match cache_service
        .delete_cached_image(user_id, &image_hash)
        .await
    {
        Ok(()) => {
            info!("Successfully deleted cached image: {}", image_hash);
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "message": "Image deleted successfully"
            })))
        }
        Err(e) => {
            error!("Failed to delete cached image {}: {:?}", image_hash, e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to delete cached image".to_string(),
            }))
        }
    }
}

async fn handle_clear_user_cache(
    user: AuthenticatedUser,
    cache_service: web::Data<CacheService>,
) -> Result<HttpResponse, Error> {
    let user_id = user.0;

    info!("Clearing all cached data for user: {}", user_id);

    match cache_service.clear_user_cache(user_id).await {
        Ok(()) => {
            info!("Successfully cleared all cached data for user: {}", user_id);
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "message": "All cached data cleared successfully"
            })))
        }
        Err(e) => {
            error!("Failed to clear cached data for user {}: {:?}", user_id, e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to clear cached data".to_string(),
            }))
        }
    }
}
