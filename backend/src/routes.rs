use actix_files::Files;
use actix_web::{web, HttpResponse, Error};
use actix_multipart::Multipart;
use serde_json::json;
use serde::Serialize;
use uuid::Uuid;
use std::io::Write;
use std::sync::{Arc, Mutex};
use log::{info, error};
use shared::{InferenceResponse, ProcessingMode};
use futures::{StreamExt, TryStreamExt};
use crate::pyprocess::model::Model;
use crate::db::task_repository::TaskRepository;

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

pub fn configure_routes(cfg: &mut web::ServiceConfig, frontend_dir: String) {
    cfg.service(web::resource("/api/inference").route(web::post().to(handle_inference)))
    .service(web::resource("/api/tasks").route(web::post().to(create_task)))
    .service(web::resource("/api/tasks/{task_id}").route(web::get().to(get_task)))
    .service(Files::new("/static", &frontend_dir).show_files_listing())
    .service(Files::new("/dist/dir", frontend_dir).index_file("index.html"));
}

async fn handle_inference(
    model: web::Data<Arc<Mutex<Model>>>,
    mut payload: Multipart,
    task_repo: web::Data<TaskRepository>
) -> Result<HttpResponse, Error> {
    let mut results = Vec::new();
    let mut images: Vec<Vec<u8>> = Vec::new();
    let mut task_ids = Vec::new();
    let mut processing_mode = ProcessingMode::IntermediateFusionEnsemble;

    while let Ok(Some(mut field)) = payload.try_next().await {
        let content_disposition = field.content_disposition();
        let field_name = content_disposition.get_name().unwrap_or("");

        match field_name {
            "image" => {
                let mut image_data = Vec::new();
                while let Some(chunk) = field.next().await {
                    image_data.write_all(&chunk?)?;
                }
                if !image_data.is_empty() {
                    images.push(image_data);
                }
            }
            "mode" => {
                let mut mode_data = Vec::new();
                while let Some(chunk) = field.next().await {
                    mode_data.write_all(&chunk?)?;
                }
                if let Ok(mode_str) = String::from_utf8(mode_data) {
                    processing_mode = match mode_str.as_str() {
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

    if images.is_empty() {
        error!("No image data received in inference request");
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "No image data received.".to_string(),
        }));
    }

    let mut model_guard = model.lock().map_err(|e| {
        error!("Failed to lock model: {:?}", e);
        actix_web::error::ErrorInternalServerError("Failed to lock model")
    })?;

    if let Err(e) = model_guard.persistent_load(&processing_mode) {
        error!("Failed to load models for mode {:?}: {:?}", processing_mode, e);
        return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Failed to load models for mode {:?}", processing_mode),
        }));
    }

    for image_data in &images {
        let task_id = Uuid::new_v4();
        task_ids.push(task_id);

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
                        "status": "pending"
                    },
                    "inference": response
                }));
            },
            Err(e) => {
                let error_msg = format!("Model inference error: {:?}", e);
                error!("{}", error_msg);

                results.push(json!({
                    "task": {
                        "id": task_id,
                        "status": "pending"
                    },
                    "error": error_msg
                }));
            }
        }
    }

    let task_repo_clone = task_repo.clone();
    let images_clone = images.clone();
    let results_clone = results.clone();
    let task_ids_clone = task_ids.clone();

    actix_web::rt::spawn(async move {
        for ((result_json, _image_data), task_id) in results_clone.into_iter().zip(images_clone).zip(task_ids_clone) {
            match task_repo_clone.create_task_with_id(task_id, None, None).await {
                Ok(_task) => {
                    if let Some(inf) = result_json.get("inference") {
                        let preds_json = inf.get("predictions").cloned().unwrap_or(json!([]));
                        task_repo_clone.update_task_result(task_id, preds_json).await.ok();
                        task_repo_clone.update_task_status(task_id, "completed").await.ok();
                    } else if let Some(err_msg) = result_json.get("error").and_then(|e| e.as_str()) {
                        task_repo_clone.update_task_error(task_id, err_msg).await.ok();
                    }
                }
                Err(e) => {
                    error!("Failed to create task with id in background: {:?}", e);
                }
            }
        }
    });

    Ok(HttpResponse::Ok().json(json!({
        "results": results
    })))
}

async fn create_task(task_repo: web::Data<TaskRepository>) -> HttpResponse {
    let task_id = Uuid::new_v4();
    match task_repo.create_task_with_id(task_id, None, None).await {
        Ok(task) => HttpResponse::Created().json(task),
        Err(e) => {
            error!("Failed to create task: {:?}", e);
            let resp = ErrorResponse { error: "Failed to create task.".into() };
            HttpResponse::InternalServerError().json(resp)
        }
    }
}

async fn get_task(task_repo: web::Data<TaskRepository>, path: web::Path<String>) -> HttpResponse {
    let task_id_str = path.into_inner();
    let task_id = match Uuid::parse_str(&task_id_str) {
        Ok(uuid) => uuid,
        Err(_) => return HttpResponse::BadRequest().body("Invalid UUID format."),
    };
    match task_repo.get_task(task_id).await {
        Ok(Some(task)) => {
            info!("Retrieved task: {}", task_id);
            HttpResponse::Ok().json(task)
        },
        Ok(None) => {
            info!("Task not yet created: {}", task_id);
            HttpResponse::Accepted().body("Logging to the DB...")
        },
        Err(e) => {
            error!("Error retrieving task {}: {:?}", task_id, e);
            HttpResponse::InternalServerError().body(format!("Error retrieving task: {:?}", e))
        }
    }
}


