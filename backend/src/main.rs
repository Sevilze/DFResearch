mod auth;
mod cache;
mod db;
mod pyprocess;
mod routes;
mod storage;

use actix_cors::Cors;
use actix_web::{web, App, HttpServer};
use auth::cognito_service::CognitoService;
use auth::jwt::JwtService;
use auth::middleware::AuthMiddleware;
use aws_config::BehaviorVersion;
use aws_sdk_cognitoidentityprovider::Client as CognitoClient;
use aws_sdk_dynamodb::Client as DynamoDbClient;
use aws_sdk_s3::Client as S3Client;
use cache::cache_service::CacheService;
use db::dynamodb_repository::DynamoDbRepository;
use pyprocess::model::Model;
use routes::configure_routes;
use std::env;
use std::sync::{Arc, Mutex};
use storage::s3_service::S3Service;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    if let Ok(current_dir) = env::current_dir() {
        log::info!("Current working directory: {}", current_dir.display());
    } else {
        log::error!("Failed to get the current working directory.");
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let frontend_dir = format!("{}/../frontend/dist", manifest_dir);
    let model = Arc::new(Mutex::new(Model::new()));

    dotenv::dotenv().ok();

    // Initialize AWS configuration
    let aws_config = aws_config::defaults(BehaviorVersion::latest()).load().await;

    // Create AWS clients
    let dynamodb_client = DynamoDbClient::new(&aws_config);
    let s3_client = S3Client::new(&aws_config);
    let cognito_client = CognitoClient::new(&aws_config);

    // Get table names from environment
    let users_table =
        env::var("DYNAMODB_USERS_TABLE").unwrap_or_else(|_| "dfresearch-users".to_string());
    let images_table =
        env::var("DYNAMODB_IMAGES_TABLE").unwrap_or_else(|_| "dfresearch-images".to_string());
    let inference_table =
        env::var("DYNAMODB_INFERENCE_TABLE").unwrap_or_else(|_| "dfresearch-inference".to_string());
    let s3_bucket = env::var("S3_BUCKET_NAME").unwrap_or_else(|_| "dfresearch-images".to_string());

    // Create repository and services
    let db_repo =
        DynamoDbRepository::new(dynamodb_client, users_table, images_table, inference_table);
    let s3_service = S3Service::new(s3_client, s3_bucket);
    let cache_service = CacheService::new(db_repo.clone(), s3_service);

    // Create authentication services
    let jwt_secret = env::var("JWT_SECRET")
        .unwrap_or_else(|_| "your-secret-key-change-in-production".to_string());
    let jwt_service = JwtService::new(&jwt_secret);

    // Cognito service
    let cognito_user_pool_id = env::var("COGNITO_USER_POOL_ID")
        .unwrap_or_else(|_| "your_cognito_user_pool_id_here".to_string());
    let cognito_client_id =
        env::var("COGNITO_CLIENT_ID").unwrap_or_else(|_| "your_cognito_client_id_here".to_string());
    let cognito_client_secret = env::var("COGNITO_CLIENT_SECRET")
        .unwrap_or_else(|_| "your_cognito_client_secret_here".to_string());
    let cognito_domain =
        env::var("COGNITO_DOMAIN").unwrap_or_else(|_| "dfresearch-auth".to_string());
    let cognito_redirect_uri = env::var("COGNITO_REDIRECT_URI")
        .unwrap_or_else(|_| "http://localhost:8081/auth/cognito/callback".to_string());
    let aws_region = env::var("AWS_REGION").unwrap_or_else(|_| "ap-southeast-2".to_string());

    // Check configuration before moving values
    let cognito_configured = !cognito_user_pool_id.contains("your_cognito")
        && !cognito_client_id.contains("your_cognito");

    let cognito_service = CognitoService::new(
        cognito_client,
        cognito_user_pool_id,
        cognito_client_id,
        cognito_client_secret,
        cognito_domain,
        cognito_redirect_uri,
        aws_region,
    );

    let auth_middleware = AuthMiddleware::new(jwt_service.clone());

    // Log authentication configuration status
    log::info!("Authentication Configuration");
    if cognito_configured {
        log::info!("Cognito configuration detected");
    } else {
        log::warn!("Cognito is not fully configured. Please run AWS setup commands and update .env file.");
    }

    log::info!("Authentication endpoint:");
    if cognito_configured {
        log::info!("Login: http://localhost:8081/auth/login");
    } else {
        log::info!("Login (needs setup): http://localhost:8081/auth/login");
    }

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
            .app_data(web::Data::new(cache_service.clone()))
            .app_data(web::Data::new(db_repo.clone()))
            .app_data(web::Data::new(jwt_service.clone()))
            .app_data(web::Data::new(cognito_service.clone()))
            .configure(|cfg| configure_routes(cfg, frontend_dir.clone(), auth_middleware.clone()))
    })
    .bind("0.0.0.0:8081")?
    .run()
    .await
}
