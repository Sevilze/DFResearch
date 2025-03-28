use actix_web::{web, HttpResponse};
use super::model::Model;

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/inference").
        route(web::post().to(handle_inference))
    );
}

async fn handle_inference(model: web::Data<Model>, image_data: web::Bytes) -> HttpResponse {
    match model.inference(&image_data) {
        Ok(results) => HttpResponse::Ok().json(results),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error : {:?}", e)),
    }
}