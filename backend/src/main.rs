mod routes;
mod ddb;
mod pyprocess;
use actix_cors::Cors;
use actix_files::Files;
use actix_web::{web, App, HttpServer, post, HttpResponse, Responder};
use pyprocess::model::Model;
use routes::configure_routes;
use ddb::task_service::TaskService;
use actix_multipart::Multipart;
use futures_util::StreamExt;
use std::io::Write;
use std::fs;
use std::env;
use shared::InferenceResponse;

#[post("/api/inference")]
async fn inference_handler(
    mut payload: Multipart,
    model: web::Data<Model>,
) -> impl Responder {
    let mut image_data = Vec::new();

    while let Some(item) = payload.next().await {
        let mut field = item.unwrap();
        while let Some(chunk) = field.next().await {
            image_data.extend_from_slice(&chunk.unwrap());
        }
    }

    if image_data.is_empty() {
        return HttpResponse::BadRequest().body("No image uploaded.");
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
            HttpResponse::Ok().json(response)
        },
        Err(err) => {
            log::error!("Inference error: {:?}", err);
            HttpResponse::InternalServerError().body("Inference failed.")
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    if let Ok(current_dir) = env::current_dir() {
        log::info!("Current working directory: {}", current_dir.display());
    } else {
        log::error!("Failed to get the current working directory.");
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let frontend_dir = format!("{}/../frontend", manifest_dir);
    let dist_dir = format!("{}/../frontend/dist", manifest_dir);
    let model_path = format!("{}/../pyproject/models/EarlyFusionEnsemble/best_model/EarlyFusionEnsemble_scripted.pt", manifest_dir); 
    let model = Model::new(&model_path);
    
    log::info!("Initializing DynamoDB task service");
    let task_service = TaskService::new("table_data".to_string())
        .await
        .expect("Failed to create TaskService");
    log::info!("DynamoDB task service initialized");

    HttpServer::new(move || {
        App::new()
            .wrap(
                Cors::default()
                    .allow_any_origin()
                    .allowed_methods(vec!["GET", "POST", "OPTIONS"])
                    .allowed_headers(vec![
                        actix_web::http::header::AUTHORIZATION,
                        actix_web::http::header::ACCEPT,
                        actix_web::http::header::CONTENT_TYPE,
                    ])
                    .max_age(3600),
            )
            .app_data(web::Data::new(model.clone()))
            .app_data(web::Data::new(task_service.clone()))
            .configure(|cfg| routes::configure_routes(cfg, frontend_dir.clone()))
            .service(inference_handler)
            .service(Files::new("/", dist_dir.clone()).index_file("index.html"))
    })
    .bind("0.0.0.0:8081")?
    .run()
    .await
}