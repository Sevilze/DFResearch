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

    let frontend_dir = if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        format!("{}/../frontend/dist", manifest_dir)
    } else {
        "/usr/src/app/frontend/dist".to_string()
    };

    let mut model = Model::new();
    if let Err(e) = model.persistent_load(&shared::ProcessingMode::IntermediateFusionEnsemble) {
        log::error!("Failed to preload model at startup: {:?}", e);
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Model loading failed: {:?}", e),
        ));
    }

    let model = Arc::new(Mutex::new(model));
    dotenv::dotenv().ok();

    // Initialize AWS configuration
    let aws_config = aws_config::defaults(BehaviorVersion::latest()).load().await;

    // Create AWS clients
    let dynamodb_client = DynamoDbClient::new(&aws_config);
    let s3_client = S3Client::new(&aws_config);
    let cognito_client = CognitoClient::new(&aws_config);

    // Get table names from environment
    let users_table = env::var("DYNAMODB_USERS_TABLE").unwrap().to_string();
    let images_table = env::var("DYNAMODB_IMAGES_TABLE").unwrap().to_string();
    let inference_table = env::var("DYNAMODB_INFERENCE_TABLE").unwrap().to_string();
    let s3_bucket = env::var("S3_BUCKET_NAME").unwrap().to_string();

    // Create repository and services
    let db_repo =
        DynamoDbRepository::new(dynamodb_client, users_table, images_table, inference_table);
    let s3_service = S3Service::new(s3_client, s3_bucket);
    let cache_service = CacheService::new(db_repo.clone(), s3_service);

    // Create authentication services
    let jwt_secret = env::var("JWT_SECRET").unwrap().to_string();
    let jwt_service = JwtService::new(&jwt_secret);

    // Cognito service
    let cognito_user_pool_id = env::var("COGNITO_USER_POOL_ID").unwrap().to_string();
    let cognito_client_id = env::var("COGNITO_CLIENT_ID").unwrap().to_string();
    let cognito_client_secret = env::var("COGNITO_CLIENT_SECRET").unwrap().to_string();
    let cognito_domain = env::var("COGNITO_DOMAIN").unwrap().to_string();
    let cognito_redirect_uri = env::var("COGNITO_REDIRECT_URI").unwrap().to_string();
    let aws_region = env::var("AWS_REGION").unwrap().to_string();

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
    if cognito_configured {
        log::info!("Cognito configuration detected");
    } else {
        log::warn!(
            "Cognito is not fully configured. Please run AWS setup commands and update .env file."
        );
    }

    let base_url = env::var("BASE_URL").unwrap().to_string();
    if cognito_configured {
        log::info!("Login: {}/auth/login", base_url);
    } else {
        log::info!("Login (needs setup): {}/auth/login", base_url);
    }

    let port = env::var("PORT").unwrap_or_else(|_| "8081".to_string());
    let bind_address = format!("0.0.0.0:{}", port);

    log::info!("Starting server on {}", bind_address);

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
    .bind(&bind_address)?
    .run()
    .await
}
