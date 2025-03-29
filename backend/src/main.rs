use actix_web::{web, App, HttpServer};
use model::Model;
use routes::configure_routes;

mod model;
mod routes;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = format!("{}/../pyproject/models/EarlyFusionEnsemble/best_model/EarlyFusionEnsemble_scripted.pt", manifest_dir);
    let model = Model::new(&model_path);

    HttpServer::new(move ||{
        App::new().
        app_data(web::Data::new(model.clone())).
        configure(configure_routes)
    })
    .bind("127.0.0.1:8081")?
    .run()
    .await
}
