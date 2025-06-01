use actix_web::{web, HttpResponse, Result};
use log::{error, info};
use serde_json::json;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::cache::models::User;
use crate::db::dynamodb_repository::DynamoDbRepository;

use super::cognito_service::CognitoService;
use super::jwt::JwtService;
use super::middleware::AuthenticatedUser;
use super::models::{AuthCallbackQuery, AuthUser};

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

// In-memory state store for OAuth CSRF protection
lazy_static::lazy_static! {
    static ref OAUTH_STATES: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
}

pub async fn cognito_login_redirect(
    cognito_service: web::Data<CognitoService>,
) -> Result<HttpResponse> {
    let state = Uuid::new_v4().to_string();

    // Store state for CSRF protection
    {
        let mut states = OAUTH_STATES.lock().unwrap();
        states.insert(state.clone());
        info!("Generated OAuth state: {}", state);
    }

    match cognito_service.get_authorization_url(&state) {
        Ok(auth_url) => {
            info!("Redirecting to Cognito authorization URL with state protection");
            Ok(HttpResponse::Found()
                .append_header(("Location", auth_url))
                .finish())
        }
        Err(e) => {
            error!("Failed to generate Cognito authorization URL: {:?}", e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to initiate authentication".to_string(),
            }))
        }
    }
}

pub async fn cognito_auth_callback(
    query: web::Query<AuthCallbackQuery>,
    cognito_service: web::Data<CognitoService>,
    jwt_service: web::Data<JwtService>,
    db_repo: web::Data<DynamoDbRepository>,
) -> Result<HttpResponse> {
    // Check for OAuth errors first
    if let Some(error) = &query.error {
        let error_desc = query
            .error_description
            .as_deref()
            .unwrap_or("Unknown error");
        error!("OAuth error received: {} - {}", error, error_desc);

        let frontend_error_url = format!(
            "http://localhost:8081/?error={}&error_description={}",
            urlencoding::encode(error),
            urlencoding::encode(error_desc)
        );

        return Ok(HttpResponse::Found()
            .append_header(("Location", frontend_error_url))
            .finish());
    }

    // Validate state parameter for CSRF protection
    if let Some(state) = &query.state {
        let mut states = OAUTH_STATES.lock().unwrap();
        if !states.remove(state) {
            error!("Invalid or expired OAuth state: {}", state);
            return Ok(HttpResponse::BadRequest().json(ErrorResponse {
                error: "Invalid authentication state. Possible CSRF attack.".to_string(),
            }));
        }
        info!("OAuth state validated and consumed: {}", state);
    } else {
        error!("No state parameter received - possible CSRF attack");
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error: "Missing authentication state parameter".to_string(),
        }));
    }

    // Extract authorization code
    let code = match &query.code {
        Some(code) => code,
        None => {
            error!("No authorization code received from Cognito");
            return Ok(HttpResponse::BadRequest().json(ErrorResponse {
                error: "No authorization code received".to_string(),
            }));
        }
    };

    // Exchange authorization code for tokens
    let token_response = match cognito_service.exchange_code_for_tokens(code).await {
        Ok(tokens) => tokens,
        Err(e) => {
            error!("Failed to exchange code for tokens: {:?}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to exchange authorization code".to_string(),
            }));
        }
    };

    // Get user info from Cognito
    let user_info = match cognito_service
        .get_user_info(&token_response.access_token)
        .await
    {
        Ok(info) => info,
        Err(e) => {
            error!("Failed to get user info from Cognito: {:?}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to get user information".to_string(),
            }));
        }
    };

    // Create or update user in database
    let user = User {
        id: Uuid::new_v4(),
        cognito_sub: user_info.sub.clone(),
        email: user_info.email.clone(),
        name: format!(
            "{} {}",
            user_info.given_name.unwrap_or_default(),
            user_info.family_name.unwrap_or_default()
        )
        .trim()
        .to_string(),
        picture_url: user_info.picture,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_login: Some(chrono::Utc::now()),
        is_active: true,
        cognito_access_token: Some(token_response.access_token),
        cognito_refresh_token: token_response.refresh_token,
    };

    // Check if user already exists by Cognito sub
    let final_user = match db_repo.get_user_by_cognito_sub(&user_info.sub).await {
        Ok(mut existing) => {
            existing.email = user.email;
            existing.name = user.name;
            existing.picture_url = user.picture_url;
            existing.update_last_login();
            existing.cognito_access_token = user.cognito_access_token;
            existing.cognito_refresh_token = user.cognito_refresh_token;

            match db_repo.update_user(&existing).await {
                Ok(_) => {
                    info!("Updated existing user: {}", existing.email);
                    existing
                }
                Err(e) => {
                    error!("Failed to update user: {:?}", e);
                    return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                        error: "Failed to update user".to_string(),
                    }));
                }
            }
        }
        Err(e) => {
            // Create new user
            info!(
                "User not found by cognito_sub, creating new user: {} (error: {:?})",
                user.email, e
            );
            match db_repo.create_user(&user).await {
                Ok(_) => {
                    info!("Successfully created new user: {}", user.email);
                    user
                }
                Err(e) => {
                    error!("Failed to create user {}: {:?}", user.email, e);
                    return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                        error: format!("Failed to create user: {:?}", e),
                    }));
                }
            }
        }
    };

    // Generate JWT token
    let auth_user = AuthUser {
        id: final_user.id,
        email: final_user.email.clone(),
        name: final_user.name.clone(),
        picture_url: final_user.picture_url.clone(),
    };

    let jwt_token = match jwt_service.generate_token(&auth_user) {
        Ok(token) => token,
        Err(e) => {
            error!("Failed to generate JWT token: {:?}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to generate authentication token".to_string(),
            }));
        }
    };

    info!(
        "User authenticated successfully via Cognito: {}",
        final_user.email
    );

    // Redirect to frontend with token as URL parameter
    let frontend_url = format!("http://localhost:8081/?token={}", jwt_token);

    Ok(HttpResponse::Found()
        .append_header(("Location", frontend_url))
        .finish())
}

pub async fn cognito_refresh_token(
    cognito_service: web::Data<CognitoService>,
    jwt_service: web::Data<JwtService>,
    db_repo: web::Data<DynamoDbRepository>,
    user: AuthenticatedUser,
) -> Result<HttpResponse> {
    let db_user = match db_repo.get_user(&user.0).await {
        Ok(user) => user,
        Err(e) => {
            error!("Failed to get user from database: {:?}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to get user information".to_string(),
            }));
        }
    };

    let refresh_token = match &db_user.cognito_refresh_token {
        Some(token) => token,
        None => {
            error!("No refresh token available for user: {}", user.0);
            return Ok(HttpResponse::BadRequest().json(ErrorResponse {
                error: "No refresh token available".to_string(),
            }));
        }
    };

    // Refresh tokens with Cognito
    let token_response = match cognito_service.refresh_token(refresh_token).await {
        Ok(tokens) => tokens,
        Err(e) => {
            error!("Failed to refresh tokens: {:?}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to refresh authentication".to_string(),
            }));
        }
    };

    // Update user with new tokens
    let mut updated_user = db_user;
    updated_user.cognito_access_token = Some(token_response.access_token);
    if let Some(new_refresh_token) = token_response.refresh_token {
        updated_user.cognito_refresh_token = Some(new_refresh_token);
    }
    updated_user.updated_at = chrono::Utc::now();

    if let Err(e) = db_repo.update_user(&updated_user).await {
        error!("Failed to update user tokens: {:?}", e);
        return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
            error: "Failed to update user tokens".to_string(),
        }));
    }

    // Generate new JWT token using refresh method
    let auth_user = AuthUser {
        id: updated_user.id,
        email: updated_user.email.clone(),
        name: updated_user.name.clone(),
        picture_url: updated_user.picture_url.clone(),
    };

    let jwt_token = match jwt_service.refresh_token(&auth_user) {
        Ok(token) => token,
        Err(e) => {
            error!("Failed to refresh JWT token: {:?}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to refresh authentication token".to_string(),
            }));
        }
    };

    Ok(HttpResponse::Ok().json(json!({ "token": jwt_token })))
}

pub async fn cognito_logout(
    cognito_service: web::Data<CognitoService>,
    db_repo: web::Data<DynamoDbRepository>,
    user: AuthenticatedUser,
) -> Result<HttpResponse> {
    let db_user = match db_repo.get_user(&user.0).await {
        Ok(user) => user,
        Err(e) => {
            error!("Failed to get user from database: {:?}", e);
            return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to get user information".to_string(),
            }));
        }
    };

    // Revoke tokens with Cognito if available
    if let Some(access_token) = &db_user.cognito_access_token {
        if let Err(e) = cognito_service.revoke_token(access_token).await {
            error!("Failed to revoke Cognito tokens: {:?}", e);
        }
    }

    // Clear tokens from database
    let mut updated_user = db_user;
    updated_user.cognito_access_token = None;
    updated_user.cognito_refresh_token = None;
    updated_user.updated_at = chrono::Utc::now();

    if let Err(e) = db_repo.update_user(&updated_user).await {
        error!("Failed to clear user tokens: {:?}", e);
        return Ok(HttpResponse::InternalServerError().json(ErrorResponse {
            error: "Failed to clear user tokens".to_string(),
        }));
    }

    info!("User logged out successfully: {}", updated_user.email);

    Ok(HttpResponse::Ok().json(json!({
        "message": "Logged out successfully"
    })))
}
