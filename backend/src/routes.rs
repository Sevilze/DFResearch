use actix_files::Files;
use actix_web::{web, HttpResponse, Responder, Error};
use actix_multipart::Multipart;
use futures::{StreamExt, TryStreamExt};
use log::{info, error};
use std::io::Write;
use crate::pyprocess::model::Model;
use super::ddb::task_service::TaskService;
use super::ddb::model::TaskStatus;
use serde::Serialize;
use shared::{InferenceRequest, InferenceResponse};

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

pub fn configure_routes(cfg: &mut web::ServiceConfig, frontend_dir: String) {
    cfg.service(web::resource("/inference").route(web::post().to(handle_inference)))
    .service(web::resource("/tasks").route(web::post().to(create_task)))
    .service(web::resource("/tasks/{task_id}").route(web::get().to(get_task)))
    .service(Files::new("/static", frontend_dir).show_files_listing());
}

async fn index() -> impl Responder {
    HttpResponse::Ok().body("Welcome to the server.")
}

async fn handle_inference(
    model: web::Data<Model>,
    mut payload: Multipart,
    task_service: web::Data<TaskService>
) -> Result<HttpResponse, Error> {
    info!("Received inference request");
    let task = match task_service.create_task().await {
        Ok(task) => task,
        Err(e) => {
            error!("Failed to create task in DynamoDB: {:?}", e);
            let resp = ErrorResponse { error: "Failed to create task".into() };
            return Ok(HttpResponse::InternalServerError().json(resp));
        }
    };

    let task_id = task.id.clone();
    info!("Created task: {}", task_id);

    if let Err(e) = task_service.update_task_status(&task_id, TaskStatus::Processing).await {
        error!("Failed to update task status: {:?}", e);
    }

    let mut image_data = Vec::new();
    while let Ok(Some(mut field)) = payload.try_next().await {
        if field.name() == "image" {
            while let Some(chunk) = field.next().await {
                let data = chunk?;
                image_data.write_all(&data)?;
            }
            break;
        }
    }

    if image_data.is_empty() {
        if let Err(e) = task_service.update_task_error(&task_id, "No image data provided".into()).await {
            error!("Failed to update task error: {:?}", e);
        }
        return Ok(HttpResponse::BadRequest().body("No image data provided"));
    }

    match model.inference(&image_data) {
        Ok(predictions) => {
            let (is_ai, confidence) = model.calculate_result(&predictions);
            let response = InferenceResponse {
                predictions,
                class_labels: vec!["AI Generated".into(), "Human Created".into()],
                is_ai,
                confidence,
            };
            info!("Inference successful for task: {}", task_id);
            let task_service_clone = task_service.clone();
            let task_id_clone = task_id.clone();
            let predictions_clone = response.predictions.clone();
            actix_web::rt::spawn(async move {
                if let Err(e) = task_service_clone.update_task_result(&task_id_clone, predictions_clone).await {
                    error!("Failed to update task with results: {:?}", e);
                } else {
                    info!("Task {} updated with results", task_id_clone);
                }
            });
            Ok(HttpResponse::Ok().json(response))
        },
        Err(e) => {
            let error_msg = format!("Model inference error: {:?}", e);
            error!("{} for task: {}", error_msg, task_id);
            let task_service_clone = task_service.clone();
            let task_id_clone = task_id.clone();
            let error_msg_clone = error_msg.clone();
            actix_web::rt::spawn(async move {
                if let Err(e) = task_service_clone.update_task_error(&task_id_clone, error_msg_clone).await {
                    error!("Failed to update task with error: {:?}", e);
                }
            });
            Ok(HttpResponse::InternalServerError().body(error_msg))
        }
    }
}

async fn create_task(task_service: web::Data<TaskService>) -> HttpResponse {
    match task_service.create_task().await {
        Ok(task) => HttpResponse::Created().json(task),
        Err(e) => {
            error!("Failed to create task in DynamoDB: {:?}", e);
            let resp = ErrorResponse { error: "Failed to create task".into() };
            HttpResponse::InternalServerError().json(resp)
        }
    }
}

async fn get_task(task_service: web::Data<TaskService>, path: web::Path<String>) -> HttpResponse {
    let task_id = path.into_inner();
    match task_service.get_task(&task_id).await {
        Ok(Some(task)) => {
            info!("Retrieved task: {}", task_id);
            HttpResponse::Ok().json(task)
        },
        Ok(None) => {
            error!("Task not found: {}", task_id);
            HttpResponse::NotFound().body(format!("Task with ID {} not found", task_id))
        },
        Err(e) => {
            error!("Error retrieving task {}: {:?}", task_id, e);
            HttpResponse::InternalServerError().body(format!("Error retrieving task: {:?}", e))
        }
    }
}
