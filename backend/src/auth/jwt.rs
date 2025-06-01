use super::models::{AuthUser, Claims};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};

#[derive(Debug, thiserror::Error)]
pub enum JwtError {
    #[error("JWT encoding error: {0}")]
    Encoding(#[from] jsonwebtoken::errors::Error),
    #[error("JWT decoding error: {0}")]
    Decoding(String),
    #[error("Invalid token")]
    InvalidToken,
    #[error("Token expired")]
    TokenExpired,
}

#[derive(Clone)]
pub struct JwtService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl JwtService {
    pub fn new(secret: &str) -> Self {
        Self {
            encoding_key: EncodingKey::from_secret(secret.as_ref()),
            decoding_key: DecodingKey::from_secret(secret.as_ref()),
        }
    }

    pub fn generate_token(&self, user: &AuthUser) -> Result<String, JwtError> {
        let now = Utc::now();
        let expiration = now + Duration::hours(24);

        let claims = Claims {
            sub: user.id.to_string(),
            email: user.email.clone(),
            name: user.name.clone(),
            exp: expiration.timestamp() as usize,
            iat: now.timestamp() as usize,
        };

        let header = Header::new(Algorithm::HS256);
        encode(&header, &claims, &self.encoding_key).map_err(JwtError::Encoding)
    }

    pub fn verify_token(&self, token: &str) -> Result<Claims, JwtError> {
        if token.is_empty() {
            return Err(JwtError::InvalidToken);
        }

        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(JwtError::InvalidToken);
        }

        let validation = Validation::new(Algorithm::HS256);

        match decode::<Claims>(token, &self.decoding_key, &validation) {
            Ok(token_data) => {
                let now = Utc::now().timestamp() as usize;
                log::debug!(
                    "JWT token decoded successfully. User: {}, Email: {}, Exp: {}, Now: {}",
                    token_data.claims.sub,
                    token_data.claims.email,
                    token_data.claims.exp,
                    now
                );

                if token_data.claims.exp < now {
                    log::warn!(
                        "JWT token expired. Exp: {}, Now: {}",
                        token_data.claims.exp,
                        now
                    );
                    return Err(JwtError::TokenExpired);
                }
                Ok(token_data.claims)
            }
            Err(err) => {
                log::error!("JWT token decode error: {:?}", err);
                match err.kind() {
                    jsonwebtoken::errors::ErrorKind::ExpiredSignature => {
                        Err(JwtError::TokenExpired)
                    }
                    jsonwebtoken::errors::ErrorKind::InvalidToken => Err(JwtError::InvalidToken),
                    jsonwebtoken::errors::ErrorKind::InvalidSignature => {
                        Err(JwtError::InvalidToken)
                    }
                    _ => Err(JwtError::Decoding(err.to_string())),
                }
            }
        }
    }

    pub fn refresh_token(&self, user: &AuthUser) -> Result<String, JwtError> {
        // Generate a new JWT token with fresh expiration time
        // This is called when the Cognito refresh token is used successfully
        self.generate_token(user)
    }
}
