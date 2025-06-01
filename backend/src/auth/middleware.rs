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
            let path = req.path();
            if path.starts_with("/static/")
                || path.starts_with("/dist/")
                || path.ends_with(".html")
                || path.ends_with(".css")
                || path.ends_with(".js")
                || path.ends_with(".wasm")
                || path.ends_with(".png")
                || path.ends_with(".jpg")
                || path.ends_with(".jpeg")
                || path.ends_with(".gif")
                || path.ends_with(".svg")
                || path.ends_with(".ico")
                || path == "/"
            {
                let res = service.call(req).await?;
                return Ok(res.map_into_left_body());
            }

            // Extract Bearer token from Authorization header
            let auth_header = req.headers().get("Authorization");
            log::debug!("Auth middleware processing path: {}", path);

            if let Some(auth_value) = auth_header {
                if let Ok(auth_str) = auth_value.to_str() {
                    if let Some(token) = auth_str.strip_prefix("Bearer ") {
                        log::debug!("Found Bearer token, verifying...");
                        match jwt_service.verify_token(token) {
                            Ok(claims) => {
                                log::debug!("JWT token verified for user: {}", claims.sub);
                                if let Ok(user_id) = Uuid::parse_str(&claims.sub) {
                                    req.extensions_mut().insert(user_id);
                                    let res = service.call(req).await?;
                                    return Ok(res.map_into_left_body());
                                } else {
                                    log::error!("Invalid UUID in JWT claims.sub: {}", claims.sub);
                                }
                            }
                            Err(e) => {
                                log::warn!("JWT token verification failed: {:?}", e);
                            }
                        }
                    } else {
                        log::warn!("Authorization header doesn't start with 'Bearer '");
                    }
                } else {
                    log::warn!("Invalid Authorization header format");
                }
            } else {
                log::debug!("No Authorization header found for path: {}", path);
            }

            // Return unauthorized response
            log::warn!("Authentication failed for path: {} - returning 401", path);
            let (req, _) = req.into_parts();
            let response = HttpResponse::Unauthorized()
                .json(serde_json::json!({"error": "Missing or invalid authorization token"}))
                .map_into_right_body();

            Ok(ServiceResponse::new(req, response))
        })
    }
}

pub struct AuthenticatedUser(pub Uuid);

impl FromRequest for AuthenticatedUser {
    type Error = actix_web::Error;
    type Future = Ready<Result<Self, Self::Error>>;

    fn from_request(req: &HttpRequest, _: &mut actix_web::dev::Payload) -> Self::Future {
        match req.extensions().get::<Uuid>() {
            Some(user_id) => ok(AuthenticatedUser(*user_id)),
            None => ok(AuthenticatedUser(Uuid::nil())),
        }
    }
}
