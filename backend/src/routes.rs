use actix_files::Files;
use actix_web::{ web, HttpResponse, Responder };
use super::model::Model;

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/inference").route(web::post().to(handle_inference)))
        .service(web::resource("/").route(web::get().to(index)))
        .service(Files::new("/static", "./frontend").show_files_listing());
}

async fn index() -> impl Responder {
    HttpResponse::Ok().body("Welcome to the server.")
}

async fn handle_inference(model: web::Data<Model>, image_data: web::Bytes) -> HttpResponse {
    match model.inference(&image_data) {
        Ok(results) => HttpResponse::Ok().json(results),
        Err(e) => HttpResponse::InternalServerError().body(format!("Error : {:?}", e)),
    }
}
