mod db;
mod pyprocess;
mod routes;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use db::task_repository::TaskRepository;
use pyprocess::model::Model;
use routes::configure_routes;
use std::env;
use std::sync::{Arc, Mutex};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    if let Ok(current_dir) = env::current_dir() {
        log::info!("Current working directory: {}", current_dir.display());
    } else {
        log::error!("Failed to get the current working directory.");
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let frontend_dir = format!("{}/../frontend", manifest_dir);
    let model = Arc::new(Mutex::new(Model::new()));

    dotenv::dotenv().ok();
    let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let db_pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .expect("Failed to connect to database");
    let task_repo = TaskRepository::new(db_pool.clone());

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
            .app_data(web::Data::new(task_repo.clone()))
            .configure(|cfg| configure_routes(cfg, frontend_dir.clone()))
    })
    .bind("0.0.0.0:8081")?
    .run()
    .await
}
