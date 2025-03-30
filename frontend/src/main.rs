use yew::prelude::*;
use gloo_net::http::Request;
use web_sys::HtmlInputElement;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;

enum Msg {
    SetFile(web_sys::File),
    SetPreview(String),
    Submit,
    InferenceResult(Vec<f32>),
    Error(String),
    ClearFile,
}

struct Model {
    file: Option<web_sys::File>,
    preview_url: Option<String>,
    results: Option<Vec<f32>>,
    loading: bool,
    error: Option<String>,
}

impl Component for Model {
    type Message = Msg;
    type Properties = ();

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            file: None,
            preview_url: None,
            results: None,
            loading: false,
            error: None,
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetFile(file) => {
                self.file = Some(file);
                true
            }
            Msg::SetPreview(url) => {
                self.preview_url = Some(url);
                true
            }
            Msg::Submit => {
                if let Some(file) = &self.file {
                    self.loading = true;
                    self.error = None;
                    self.results = None;
                    let file = file.clone();

                    spawn_local({
                        let link = ctx.link().clone();

                        async move {
                            let form_data = web_sys::FormData::new().unwrap();
                            form_data.append_with_blob("image", &file).unwrap();

                            let request = Request::post("/api/inference")
                                .body(form_data)
                                .expect("Failed to build request.");

                            match request.send().await {
                                Ok(response) => {
                                    if response.ok() {
                                        match response.json::<Vec<f32>>().await {
                                            Ok(results) => link.send_message(Msg::InferenceResult(results)),
                                            Err(e) => link.send_message(Msg::Error(format!("Failed to parse response: {}", e))),
                                        }
                                    } else {
                                        link.send_message(Msg::Error(format!("Server error: {}", response.status())))
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
                self.error = Some(err.clone());
                self.loading = false;
                log::error!("Error: {}", err);
                true
            }
            Msg::ClearFile => {
                self.file = None;
                self.preview_url = None;
                self.results = None;
                self.error = None;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div class="container">
                <header class="app-header">
                    <h1><i class="fa-solid fa-eye"></i> {"AI Image Detector"}</h1>
                    <p class="subtitle">{"Upload an image to check if it was generated by AI"}</p>
                </header>
                
                <main class="main-content">
                    <div class="upload-section">
                        {
                            if self.preview_url.is_none() {
                                html! {
                                    <label for="file-upload" class="upload-area">
                                        <div class="upload-placeholder">
                                            <i class="fa-solid fa-cloud-arrow-up"></i>
                                            <p>{"Drag & drop your image here or click to browse"}</p>
                                            <p class="file-types">{"Supported formats: JPG, PNG, WEBP"}</p>
                                        </div>
                                        <input 
                                            id="file-upload" 
                                            type="file" 
                                            accept="image/*" 
                                            onchange={{
                                                let link = ctx.link().clone();
                                                Callback::from(move |e: Event| {
                                                    let input: HtmlInputElement = e.target_unchecked_into();
                                                    if let Some(file_list) = input.files() {
                                                        if let Some(file) = file_list.get(0) {
                                                            if let Ok(url) = web_sys::Url::create_object_url_with_blob(&file) {
                                                                let file_clone = file.clone();
                                                                link.send_message(Msg::SetFile(file_clone));
                                                                link.send_message(Msg::SetPreview(url));
                                                                return;
                                                            }
                                                        }
                                                    }
                                                    link.send_message(Msg::Error("No file selected".into()));
                                                })
                                            }}
                                        />
                                    </label>
                                }
                            } else {
                                html! {
                                    <div class="preview-container">
                                        <div class="image-preview">
                                            <img src={self.preview_url.clone().unwrap()} alt="Preview" />
                                            <button class="clear-btn" onclick={ctx.link().callback(|_| Msg::ClearFile)}>
                                                <i class="fa-solid fa-xmark"></i>
                                            </button>
                                        </div>
                                        <div class="file-info">
                                            <p>{ self.file.as_ref().map(|f| f.name()).unwrap_or_default() }</p>
                                        </div>
                                        <button 
                                            class="analyze-btn"
                                            onclick={ctx.link().callback(|_| Msg::Submit)} 
                                            disabled={self.file.is_none() || self.loading}
                                        >
                                            { 
                                                if self.loading { 
                                                    html! { <><i class="fa-solid fa-spinner fa-spin"></i>{" Analyzing..."}</> } 
                                                } else { 
                                                    html! { <><i class="fa-solid fa-magnifying-glass"></i>{" Analyze Image"}</> } 
                                                } 
                                            }
                                        </button>
                                    </div>
                                }
                            }
                        }
                        
                        {
                            if self.preview_url.is_none() {
                                html! {
                                    <button 
                                        class="analyze-btn"
                                        onclick={ctx.link().callback(|_| Msg::Submit)} 
                                        disabled={self.file.is_none() || self.loading}
                                    >
                                        { 
                                            if self.loading { 
                                                html! { <><i class="fa-solid fa-spinner fa-spin"></i>{" Analyzing..."}</> } 
                                            } else { 
                                                html! { <><i class="fa-solid fa-magnifying-glass"></i>{" Analyze Image"}</> } 
                                            } 
                                        }
                                    </button>
                                }
                            } else {
                                html! {}
                            }
                        }
                    </div>

                    { 
                        if let Some(error_msg) = &self.error {
                            html! {
                                <div class="error-message">
                                    <i class="fa-solid fa-circle-exclamation"></i>
                                    <p>{ error_msg }</p>
                                </div>
                            }
                        } else {
                            html! {}
                        }
                    }

                    { self.view_results() }
                </main>
                
                <footer class="app-footer">
                    <p>{"AI Image Detector | Fullstack Rust WASM"}</p>
                </footer>
            </div>
        }
    }
}

impl Model {
    fn view_results(&self) -> Html {
        if let Some(results) = &self.results {
            let (max_idx, _) = results.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &0.0));
            
            let percentages: Vec<f32> = results.iter().map(|&v| v * 100.0).collect();
            
            let ai_score = if results.len() > 0 { percentages[0] } else { 0.0 };
            let is_ai = ai_score > 50.0;

            html! {
                <div class={classes!("results-container", if is_ai { "ai-detected" } else { "not-ai" })}>
                    <div class="result-header">
                        <h2>
                            { 
                                if is_ai {
                                    html! { <><i class="fa-solid fa-robot"></i>{" AI Generated"}</> }
                                } else {
                                    html! { <><i class="fa-solid fa-camera"></i>{" Likely Authentic"}</> }
                                }
                            }
                        </h2>
                        <div class="confidence-meter">
                            <div class="meter-label">{"Confidence:"}</div>
                            <div class="meter">
                                <div class="meter-fill" style={format!("width: {}%", percentages[max_idx])}></div>
                            </div>
                            <div class="meter-value">{format!("{:.1}%", percentages[max_idx])}</div>
                        </div>
                    </div>
                    
                    <div class="detailed-results">
                        <h3>{"Detailed Analysis"}</h3>
                        <div class="result-bars">
                            { for results.iter().enumerate().map(|(i, &v)| {
                                let class_name = match i {
                                    0 => "AI Generated",
                                    1 => "Human Created",
                                    _ => "Unknown Class",
                                };
                                let percentage = v * 100.0;
                                
                                html! {
                                    <div class="result-item">
                                        <div class="result-label">{ class_name }</div>
                                        <div class="result-bar-container">
                                            <div class="result-bar" style={format!("width: {}%", percentage)}></div>
                                        </div>
                                        <div class="result-value">{ format!("{:.1}%", percentage) }</div>
                                    </div>
                                }
                            })}
                        </div>
                    </div>
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