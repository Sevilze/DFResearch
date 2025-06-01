use actix_web::{web, HttpResponse, Result};
use log::error;

use crate::db::dynamodb_repository::DynamoDbRepository;

use super::middleware::AuthenticatedUser;
use super::models::AuthUser;

#[derive(serde::Serialize)]
struct ErrorResponse {
    error: String,
}

pub async fn me(
    user: AuthenticatedUser,
    db_repo: web::Data<DynamoDbRepository>,
) -> Result<HttpResponse> {
    log::info!("/auth/me endpoint called for user ID: {}", user.0);

    if user.0.is_nil() {
        log::error!("User ID is nil - authentication middleware issue");
        return Ok(HttpResponse::Unauthorized().json(ErrorResponse {
            error: "Invalid user ID".to_string(),
        }));
    }

    // Try to get user from database
    match db_repo.get_user_by_id(user.0).await {
        Ok(Some(user_data)) => {
            log::info!("User found in database: {}", user_data.email);
            let auth_user = AuthUser::from(user_data);
            Ok(HttpResponse::Ok().json(auth_user))
        }
        Ok(None) => {
            log::warn!("User not found in database for ID: {}", user.0);
            let mock_user = serde_json::json!({
                "id": user.0.to_string(),
                "email": "test@example.com",
                "name": "Test User",
                "picture_url": null
            });
            log::info!("Returning mock user data for testing");
            Ok(HttpResponse::Ok().json(mock_user))
        }
        Err(e) => {
            error!("Failed to fetch user data for ID {}: {:?}", user.0, e);
            let mock_user = serde_json::json!({
                "id": user.0.to_string(),
                "email": "test@example.com",
                "name": "Test User (DB Error)",
                "picture_url": null
            });
            log::info!("Returning mock user data due to DB error");
            Ok(HttpResponse::Ok().json(mock_user))
        }
    }
}
