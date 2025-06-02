use super::jwt::JwtService;
use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    Error, HttpMessage, HttpResponse,
};
use actix_web::{FromRequest, HttpRequest};
use futures::future::{ok, Ready};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Clone)]
pub struct AuthMiddleware {
    jwt_service: Arc<JwtService>,
}

impl AuthMiddleware {
    pub fn new(jwt_service: JwtService) -> Self {
        Self {
            jwt_service: Arc::new(jwt_service),
        }
    }
}

impl<S, B> Transform<S, ServiceRequest> for AuthMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<actix_web::body::EitherBody<B>>;
    type Error = Error;
    type Transform = AuthMiddlewareService<S>;
    type InitError = ();
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ok(AuthMiddlewareService {
            service: Arc::new(service),
            jwt_service: self.jwt_service.clone(),
        })
    }
}

pub struct AuthMiddlewareService<S> {
    service: Arc<S>,
    jwt_service: Arc<JwtService>,
}

#[derive(Debug)]
enum AuthError {
    NoAuthHeader,
    InvalidHeaderFormat,
    NotBearerToken,
    VerificationFailed(String),
    InvalidUuidInClaims(String),
}

impl AuthError {
    fn log_message(&self, path: &str) -> String {
        match self {
            AuthError::NoAuthHeader => format!("No Authorization header found for path: {}", path),
            AuthError::InvalidHeaderFormat => format!("Invalid Authorization header format (non-UTF-8) for path: {}", path),
            AuthError::NotBearerToken => format!("Authorization header for path {} doesn't start with 'Bearer '", path),
            AuthError::VerificationFailed(e) => format!("JWT token verification failed for path {}: {}", path, e),
            AuthError::InvalidUuidInClaims(sub) => format!("Invalid UUID in JWT claims.sub for path {}: {}", path, sub),
        }
    }

    fn client_error_json(&self) -> serde_json::Value {
        let error_message = match self {
            AuthError::InvalidUuidInClaims(_) => "Invalid token claims",
            AuthError::VerificationFailed(_) => "Token verification failed",
            _ => "Missing or invalid authorization token",
        };
        serde_json::json!({"error": error_message})
    }
}

/// Helper function to validate the token from the request.
fn validate_request_token(
    req: &ServiceRequest,
    jwt_service: &JwtService,
) -> Result<Uuid, AuthError> {
    let auth_header = req.headers().get("Authorization").ok_or(AuthError::NoAuthHeader)?;
    let auth_str = auth_header.to_str().map_err(|_| AuthError::InvalidHeaderFormat)?;
    let token = auth_str.strip_prefix("Bearer ").ok_or(AuthError::NotBearerToken)?;

    log::debug!("Found Bearer token, verifying...");
    let claims = jwt_service
        .verify_token(token)
        .map_err(|e| AuthError::VerificationFailed(format!("{:?}", e)))?;

    log::debug!("JWT token verified for user: {}", claims.sub);
    Uuid::parse_str(&claims.sub)
        .map_err(|_| AuthError::InvalidUuidInClaims(claims.sub.clone()))
}

impl<S, B> Service<ServiceRequest> for AuthMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<actix_web::body::EitherBody<B>>;
    type Error = Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let service = self.service.clone();
        let jwt_service = self.jwt_service.clone();

        Box::pin(async move {
            let path_str = req.path().to_string();

            if path_str.starts_with("/static/")
                || path_str.starts_with("/dist/")
                || path_str.ends_with(".html")
                || path_str.ends_with(".css")
                || path_str.ends_with(".js")
                || path_str.ends_with(".wasm")
                || path_str.ends_with(".png")
                || path_str.ends_with(".jpg")
                || path_str.ends_with(".jpeg")
                || path_str.ends_with(".gif")
                || path_str.ends_with(".svg")
                || path_str.ends_with(".ico")
                || path_str == "/"
            {
                let res = service.call(req).await?;
                return Ok(res.map_into_left_body());
            }
            log::debug!("Auth middleware processing path: {}", &path_str);

            match validate_request_token(&req, &jwt_service) {
                Ok(user_id) => {
                    req.extensions_mut().insert(user_id);
                    let res = service.call(req).await?;
                    Ok(res.map_into_left_body())
                }
                Err(auth_error) => {
                    log::warn!("{}", auth_error.log_message(&path_str));

                    let (http_req, _payload) = req.into_parts();
                    let response = HttpResponse::Unauthorized()
                        .json(auth_error.client_error_json())
                        .map_into_right_body();
                    Ok(ServiceResponse::new(http_req, response))
                }
            }
        })
    }
}

pub struct AuthenticatedUser(pub Uuid);

impl FromRequest for AuthenticatedUser {
    type Error = actix_web::Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _payload: &mut actix_web::dev::Payload) -> Self::Future {
        match req.extensions().get::<Uuid>() {
            Some(user_id) => ok(AuthenticatedUser(*user_id)),
            None => {
                // This case should ideally not be hit for routes protected by this middleware as the middleware would return 401 earlier.
                // If it's hit, it might indicate an optional authentication or misconfiguration.
                log::warn!(
                    "AuthenticatedUser extractor: No Uuid found in request extensions for path: {}. \
                    This might indicate an issue if authentication was expected.",
                    req.path()
                );
                ok(AuthenticatedUser(Uuid::nil()))
            }
        }
    }
}