use actix_web::{web, App, HttpServer};
use model::Model;
use routes::configure_routes;

mod model;
mod routes;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = Model::new("pyproject/models/EarlyFusionEnsemble/best_model/EarlyFusionEnsemble_best.pth");

    HttpServer::new(move ||{
        App::new().
        app_data(web::Data::new(model.clone())).
        configure(configure_routes)
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
