use crate::auth::middleware::{AuthMiddleware, AuthenticatedUser};
use crate::cache::cache_service::CacheService;
use crate::pyprocess::model::Model;
use crate::storage::s3_service::S3Service;
use actix_files::Files;
use actix_multipart::Multipart;
use actix_web::{web, Error, HttpResponse};
use futures::{StreamExt, TryStreamExt};
use log::{error, info, warn};
use serde::Serialize;
use serde_json::json;
use shared::{InferenceResponse, ProcessingMode};
use std::io::Write;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

pub fn configure_routes(
    cfg: &mut web::ServiceConfig,
    frontend_dir: String,
    auth_middleware: AuthMiddleware,
) {
    cfg
        // Public API endpoints (no authentication required)
        .service(web::scope("/public").route("/inference", web::post().to(handle_inference_public)))
        // Protected API endpoints (authentication required)
        .service(
            web::scope("/api")
                .wrap(auth_middleware.clone())
                .route("/inference", web::post().to(handle_inference))
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

// Handler for SPA routing - serves index.html for all unmatched routes
async fn spa_handler() -> Result<actix_files::NamedFile, actix_web::Error> {
    let frontend_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let frontend_dir = format!("{}/../frontend/dist", frontend_dir);
    let index_path = format!("{}/index.html", frontend_dir);

    actix_files::NamedFile::open(index_path).map_err(|e| {
        actix_web::error::ErrorInternalServerError(format!("Failed to serve index.html: {}", e))
    })
}

async fn handle_inference(
    model: web::Data<Arc<Mutex<Model>>>,
    mut payload: Multipart,
    cache_service: web::Data<CacheService>,
    user: AuthenticatedUser,
) -> Result<HttpResponse, Error> {
    let user_id = user.0;
    info!("Processing inference request for user: {}", user_id);

    let mut results = Vec::new();
    let mut images_with_metadata: Vec<(Vec<u8>, String, String)> = Vec::new();
    let mut processing_mode = ProcessingMode::IntermediateFusionEnsemble;

    while let Ok(Some(mut field)) = payload.try_next().await {
        let content_disposition = field.content_disposition();
        let field_name = content_disposition
            .as_ref()
            .and_then(|cd| cd.get_name())
            .unwrap_or("image");

        match field_name {
            "image" => {
                let filename = content_disposition
                    .as_ref()
                    .and_then(|cd| cd.get_filename())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "uploaded_image".to_string());

                let mut image_data = Vec::new();
                while let Some(chunk) = field.next().await {
                    image_data.write_all(&chunk?)?;
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
                    images_with_metadata.push((image_data, filename, mime_type));
                }
            }
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
                } else {
                    error!("Failed to parse processing mode string");
                }
            }
            _ => {
                info!("Ignoring unknown multipart field: {}", field_name);
            }
        }
    }

    if images_with_metadata.is_empty() {
        error!("No image data received in inference request");
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "No image data received.".to_string(),
        }));
    }

    info!(
        "Processing {} images with mode: {:?}",
        images_with_metadata.len(),
        processing_mode
    );

    for (image_data, filename, mime_type) in &images_with_metadata {
        let task_id = Uuid::new_v4();

        let image_hash = S3Service::calculate_image_hash(image_data);
        info!("Processing image '{}' with hash: {}", filename, image_hash);

        match cache_service
            .get_cached_inference(user_id, &image_hash, &processing_mode)
            .await
        {
            Ok(cached_response) => {
                info!("Cache hit for image hash: {}", image_hash);
                results.push(json!({
                    "task": { "id": task_id.to_string(), "status": "completed" },
                    "inference": cached_response,
                    "cached": true
                }));
                continue;
            }
            Err(_) => {
                info!(
                    "Cache miss for image hash: {}, proceeding with inference",
                    image_hash
                );
            }
        }

        match cache_service
            .cache_image(user_id, image_data, filename.clone(), mime_type.clone())
            .await
        {
            Ok(image_cache_entry) => {
                info!(
                    "Image '{}' cached successfully: {}",
                    filename, image_cache_entry.s3_key
                );
            }
            Err(e) => {
                warn!("Failed to cache image '{}': {:?}", filename, e);
            }
        }

        let model_operation_outcome: Result<(Vec<f32>, bool, f32), String> = {
            let mut model_guard = model.lock().map_err(|e| {
                error!("Failed to lock model: {:?}", e);
                actix_web::error::ErrorInternalServerError(format!(
                    "Critical error: Failed to lock model: {}",
                    e
                ))
            })?;

            if let Err(e) = model_guard.persistent_load(&processing_mode) {
                let error_msg = format!(
                    "Failed to load models for mode {:?}: {:?}",
                    processing_mode, e
                );
                error!("{}", error_msg);
                return Ok(
                    HttpResponse::InternalServerError().json(ErrorResponse { error: error_msg })
                );
            }

            match model_guard.inference(image_data, &processing_mode) {
                Ok(predictions_from_model) => {
                    let (is_ai, confidence) = model_guard.calculate_result(&predictions_from_model);
                    Ok((predictions_from_model.clone(), is_ai, confidence))
                }
                Err(e) => {
                    let error_msg =
                        format!("Model inference error for image '{}': {:?}", filename, e);
                    error!("{}", error_msg);
                    Err(error_msg)
                }
            }
        };

        match model_operation_outcome {
            Ok((predictions_cloned, is_ai_val, confidence_val)) => {
                let inference_result_obj = InferenceResponse {
                    predictions: predictions_cloned.clone(),
                    class_labels: vec!["AI Generated".into(), "Human Created".into()],
                    is_ai: is_ai_val,
                    confidence: confidence_val,
                };

                match cache_service
                    .cache_inference_result(
                        user_id,
                        image_hash.clone(),
                        &processing_mode,
                        &predictions_cloned,
                        is_ai_val,
                        confidence_val,
                    )
                    .await
                {
                    Ok(_) => {
                        info!("Inference result cached for image hash: {}", image_hash);
                    }
                    Err(e) => {
                        warn!(
                            "Failed to cache inference result for image hash {}: {:?}",
                            image_hash, e
                        );
                    }
                }

                results.push(json!({
                    "task": { "id": task_id.to_string(), "status": "completed" },
                    "inference": inference_result_obj,
                    "cached": false
                }));
            }
            Err(error_individual_image) => {
                results.push(json!({
                    "task": { "id": task_id.to_string(), "status": "error" },
                    "error": error_individual_image
                }));
            }
        }
    }

    info!(
        "Completed processing {} images for user: {}",
        images_with_metadata.len(),
        user_id
    );
    Ok(HttpResponse::Ok().json(json!({ "results": results })))
}

async fn handle_inference_public(
    model: web::Data<Arc<Mutex<Model>>>,
    mut payload: Multipart,
) -> Result<HttpResponse, Error> {
    info!("Processing public inference request (no authentication)");

    let mut results = Vec::new();
    let mut images_with_metadata: Vec<(Vec<u8>, String, String)> = Vec::new();
    let mut processing_mode = ProcessingMode::IntermediateFusionEnsemble;

    while let Ok(Some(mut field)) = payload.try_next().await {
        let content_disposition = field.content_disposition();
        let field_name = content_disposition
            .as_ref()
            .unwrap()
            .get_name()
            .unwrap_or("image");

        if field_name == "processing_mode" {
            let mut mode_data = Vec::new();
            while let Some(chunk) = field.next().await {
                let data = chunk?;
                mode_data.extend_from_slice(&data);
            }
            let mode_str = String::from_utf8_lossy(&mode_data);
            processing_mode = match mode_str.trim() {
                "IntermediateFusionEnsemble" => ProcessingMode::IntermediateFusionEnsemble,
                "LateFusionEnsemble" => ProcessingMode::LateFusionEnsemble,
                _ => ProcessingMode::IntermediateFusionEnsemble,
            };
            info!("Processing mode set to: {:?}", processing_mode);
            continue;
        }

        let filename = content_disposition
            .as_ref()
            .and_then(|cd| cd.get_filename())
            .unwrap_or("unknown")
            .to_string();

        let content_type = field
            .content_type()
            .map(|ct| ct.to_string())
            .unwrap_or_else(|| "application/octet-stream".to_string());

        let mut image_data = Vec::new();
        while let Some(chunk) = field.next().await {
            let data = chunk?;
            image_data.extend_from_slice(&data);
        }

        if !image_data.is_empty() {
            images_with_metadata.push((image_data, filename, content_type));
        }
    }

    info!(
        "Processing {} images with mode: {:?}",
        images_with_metadata.len(),
        processing_mode
    );

    // Process each image without caching
    for (image_data, filename, _mime_type) in &images_with_metadata {
        let task_id = Uuid::new_v4();
        info!("Processing public image: {}", filename);

        let mut model_guard = model.lock().map_err(|e| {
            error!("Failed to lock model: {:?}", e);
            actix_web::error::ErrorInternalServerError("Failed to lock model")
        })?;

        if let Err(e) = model_guard.persistent_load(&processing_mode) {
            error!(
                "Failed to load models for mode {:?}: {:?}",
                processing_mode, e
            );
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to load models for mode {:?}", processing_mode),
            }));
        }

        match model_guard.inference(image_data, &processing_mode) {
            Ok(predictions) => {
                let (is_ai, confidence) = model_guard.calculate_result(&predictions);
                let response = InferenceResponse {
                    predictions: predictions.clone(),
                    class_labels: vec!["AI Generated".into(), "Human Created".into()],
                    is_ai,
                    confidence,
                };

                results.push(json!({
                    "task": {
                        "id": task_id,
                        "status": "completed"
                    },
                    "inference": response,
                    "cached": false
                }));
            }
            Err(e) => {
                let error_msg = format!("Model inference error: {:?}", e);
                error!("{}", error_msg);

                results.push(json!({
                    "task": {
                        "id": task_id,
                        "status": "error"
                    },
                    "error": error_msg
                }));
            }
        }
    }

    info!("Completed processing {} public images", results.len());
    Ok(HttpResponse::Ok().json(json!({ "results": results })))
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
                            "confidence": latest_result.confidence
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
