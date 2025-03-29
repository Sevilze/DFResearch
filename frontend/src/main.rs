use yew::prelude::*;
use gloo_net::http::Request;
use web_sys::HtmlInputElement;
use wasm_bindgen_futures::spawn_local;

enum Msg {
    SetFile(web_sys::File),
    Submit,
    InferenceResult(Vec<f32>),
    Error(String),
}

struct Model {
    file: Option<web_sys::File>,
    results: Option<Vec<f32>>,
    loading: bool,
}

impl Component for Model {
    type Message = Msg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            file: None,
            results: None,
            loading: false,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetFile(file) => {
                self.file = Some(file);
                true
            }
            Msg::Submit => {
                if let Some(file) = &self.file {
                    self.loading = true;
                    let file = file.clone();

                    spawn_local({
                        let link = ctx.link().clone();

                        async move {
                            let form_data = web_sys::FormData::new().unwrap();
                            form_data.append_with_blob("image", &file).unwrap();

                            let request = Request::post("/inference")
                            .header("Content-Type", "multipart/form-data")
                            .body(form_data)
                            .expect("Failed to build request.");
                        
                            match request.send().await{
                                Ok(response) => {
                                    if let Ok(results) = response.json().await {
                                        link.send_message(Msg::InferenceResult(results))
                                    }
                                }
                                Err(e) => link.send_message(Msg::Error(e.to_string())),
                            }
                        }
                    });
                }
                true
            }
            Msg::InferenceResult(results) => {
                self.results = Some(results);
                self.loading = false;
                true
            }
            Msg::Error(err) => {
                self.loading = false;
                log::error!("Error: {}", err);
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let on_file_change = ctx.link().callback(|e: Event| {
            let input = e.target_unchecked_into::<HtmlInputElement>();
            if let Some(file) = input.files().and_then(|f| f.item(0)) {
                Msg::SetFile(file)
            } else {
                Msg::Error("No file selected".into())
            }
        });

        html! {
            <div class="container">
                <h1>{"ML Inference App"}</h1>
                
                <div class="file-upload">
                    <input type="file" onchange={on_file_change} />
                    <button 
                        onclick={ctx.link().callback(|_| Msg::Submit)} 
                        disabled={self.file.is_none() || self.loading}
                    >
                        { if self.loading { "Processing..." } else { "Run Inference" } }
                    </button>
                </div>

                { self.view_results() }
            </div>
        }
    }
}

impl Model {
    fn view_results(&self) -> Html {
        if let Some(results) = &self.results {
            html! {
                <div class="results">
                    <h2>{"Inference Results:"}</h2>
                    <ul>
                        { for results.iter().enumerate().map(|(i, &v)| 
                            html! { <li key={i.to_string()}>{ format!("Class {}: {:.4}", i, v) }</li> }
                        )}
                    </ul>
                </div>
            }
        } else {
            html! {}
        }
    }
}

fn main() {
    yew::Renderer::<Model>::new().render();
}
