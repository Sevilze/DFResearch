use gloo_events::EventListener;
use gloo_file::{ File as GlooFile, ObjectUrl };
use gloo_net::http::Request;
use js_sys::Date;
use shared::InferenceResponse;
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use web_sys::{ ClipboardEvent, DragEvent, FileList, HtmlInputElement };
use yew::prelude::*;

// Models
#[derive(Clone)]
struct FileData {
    id: u64,
    file: GlooFile,
    preview_url: Option<ObjectUrl>,
}

// Yew msg components
enum Msg {
    // File operations
    FilesAdded(Vec<GlooFile>),
    AddPreview(u64, ObjectUrl),
    RemoveFile(u64),
    SelectFile(u64),
    ClearAllFiles,

    // Analysis operations
    AnalyzeSelected,
    AnalyzeAll,
    InferenceResult(u64, InferenceResponse),

    // UI states
    SetError(Option<String>),
    SetDragging(bool),
    ToggleTheme,

    // Input events
    HandleDrop(DragEvent),
    HandlePaste(ClipboardEvent),
}

// Main component
struct Model {
    files: HashMap<u64, FileData>,
    selected_file_id: Option<u64>,
    results: HashMap<u64, InferenceResponse>,
    loading: bool,
    error: Option<String>,
    is_dragging: bool,
    paste_listener: Option<EventListener>,
    theme: String,
}

// Helper functions
fn generate_id() -> u64 {
    (Date::new_0().get_time() * 1000.0 + js_sys::Math::random() * 1000.0) as u64
}

// Yew component implementation
impl Component for Model {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let mut model = Self {
            files: HashMap::new(),
            selected_file_id: None,
            results: HashMap::new(),
            loading: false,
            error: None,
            is_dragging: false,
            paste_listener: None,
            theme: "light".to_string(),
        };

        let link = ctx.link().clone();
        let window = web_sys::window().expect("no global `window` exists");
        let listener = EventListener::new(&window, "paste", move |event| {
            if let Some(clipboard_event) = event.dyn_ref::<ClipboardEvent>() {
                link.send_message(Msg::HandlePaste(clipboard_event.clone()));
            }
        });
        model.paste_listener = Some(listener);

        model
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            // File operations
            Msg::FilesAdded(files) => self.handle_files_added(ctx, files),
            Msg::AddPreview(id, url) => self.handle_add_preview(id, url),
            Msg::RemoveFile(id) => self.handle_remove_file(id),
            Msg::SelectFile(id) => self.handle_select_file(id),
            Msg::ClearAllFiles => self.handle_clear_all_files(),

            // Analysis operations
            Msg::AnalyzeSelected => self.handle_analyze_selected(ctx),
            Msg::AnalyzeAll => self.handle_analyze_all(ctx),
            Msg::InferenceResult(file_id, response) => {
                self.handle_inference_result(file_id, response)
            }

            // UI states
            Msg::SetError(error) => {
                self.error = error;
                self.loading = false;
                true
            }
            Msg::SetDragging(is_dragging) => {
                if self.is_dragging != is_dragging {
                    self.is_dragging = is_dragging;
                    true
                } else {
                    false
                }
            }
            Msg::ToggleTheme => self.handle_toggle_theme(),

            // Input events
            Msg::HandleDrop(event) => self.handle_drop(ctx, event),
            Msg::HandlePaste(event) => self.handle_paste(ctx, event),
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div class="container">
                { self.render_header() }
                { self.render_theme_toggle(ctx) }

                <main class="main-content">
                    { self.render_upload_section(ctx) }
                    { self.render_error_message() }
                    { self.render_results() }
                </main>

                <footer class="app-footer">
                    <p>{"Multi-Image Upload Demo | Fullstack Rust WASM"}</p>
                </footer>
            </div>
        }
    }
}

// Handler nethods
impl Model {
    fn handle_files_added(&mut self, ctx: &Context<Self>, files: Vec<GlooFile>) -> bool {
        let current_count = self.files.len();
        let available_slots = (15usize).saturating_sub(current_count);

        if files.len() > available_slots {
            self.error = Some(
                format!("Upload limit exceeded. You can only add {} more images.", available_slots)
            );
            return true;
        }

        self.error = None;
        let mut new_selection = None;

        for file in files.into_iter() {
            let id = generate_id();
            let file_data = FileData {
                id,
                file: file.clone(),
                preview_url: None,
            };
            self.files.insert(id, file_data);

            let preview_url = ObjectUrl::from(file);
            ctx.link().send_message(Msg::AddPreview(id, preview_url));

            new_selection = Some(id);
        }

        if let Some(id) = new_selection {
            self.selected_file_id = Some(id);
        }

        true
    }

    fn handle_add_preview(&mut self, id: u64, url: ObjectUrl) -> bool {
        if let Some(file_data) = self.files.get_mut(&id) {
            file_data.preview_url = Some(url);
            true
        } else {
            false
        }
    }

    fn handle_remove_file(&mut self, id: u64) -> bool {
        if let Some(removed_file) = self.files.remove(&id) {
            drop(removed_file);
            self.results.remove(&id);

            if self.selected_file_id == Some(id) {
                self.selected_file_id = None;
            }

            if self.files.is_empty() {
                self.selected_file_id = None;
                self.results.clear();
                self.error = None;
            } else if self.selected_file_id.is_none() {
                self.selected_file_id = self.files.keys().last().cloned();
            }

            true
        } else {
            false
        }
    }

    fn handle_select_file(&mut self, id: u64) -> bool {
        if self.files.contains_key(&id) && self.selected_file_id != Some(id) {
            self.selected_file_id = Some(id);
            self.error = None;
            true
        } else {
            false
        }
    }

    fn handle_clear_all_files(&mut self) -> bool {
        for (_, file_data) in self.files.iter_mut() {
            let _ = file_data.preview_url.take();
        }
        self.files.clear();
        self.selected_file_id = None;
        self.results.clear();
        self.error = None;
        true
    }

    fn handle_analyze_selected(&mut self, ctx: &Context<Self>) -> bool {
        if let Some(file_id) = self.selected_file_id {
            if let Some(file_data) = self.files.get(&file_id) {
                self.loading = true;
                self.error = None;
                let file = file_data.file.clone();

                self.send_analysis_request(ctx, file_id, file);
                return true;
            }
        }

        ctx.link().send_message(Msg::SetError(Some("No file selected for analysis.".into())));
        false
    }

    fn handle_analyze_all(&mut self, ctx: &Context<Self>) -> bool {
        if self.files.len() > 15 {
            ctx.link().send_message(
                Msg::SetError(Some("Upload limit exceeded. Maximum of 15 images allowed.".into()))
            );
            return false;
        }

        self.loading = true;
        self.error = None;

        for file_data in self.files.values().cloned().collect::<Vec<_>>() {
            let file = file_data.file.clone();
            let file_id = file_data.id;
            self.send_analysis_request(ctx, file_id, file);
        }

        true
    }

    fn handle_inference_result(&mut self, file_id: u64, response: InferenceResponse) -> bool {
        self.results.insert(file_id, response);
        self.loading = false;
        true
    }

    fn handle_toggle_theme(&mut self) -> bool {
        let body = web_sys::window().unwrap().document().unwrap().body().unwrap();

        if self.theme == "light" {
            self.theme = "dark".to_string();
            body.class_list().add_1("dark-mode").unwrap();
        } else {
            self.theme = "light".to_string();
            body.class_list().remove_1("dark-mode").unwrap();
        }

        true
    }

    fn handle_drop(&mut self, ctx: &Context<Self>, event: DragEvent) -> bool {
        event.prevent_default();
        self.is_dragging = false;

        if let Some(data_transfer) = event.data_transfer() {
            if let Some(file_list) = data_transfer.files() {
                self.process_file_list(ctx, file_list);
            }
        }

        true
    }

    fn handle_paste(&mut self, ctx: &Context<Self>, event: ClipboardEvent) -> bool {
        if let Some(data_transfer) = event.clipboard_data() {
            if let Some(file_list) = data_transfer.files() {
                if file_list.length() > 0 {
                    event.prevent_default();
                    self.process_file_list(ctx, file_list);
                    return true;
                }
            }
        }

        false
    }

    // Helper methods
    fn process_file_list(&self, ctx: &Context<Self>, file_list: FileList) {
        let mut files_to_process = Vec::new();

        for i in 0..file_list.length() {
            if let Some(file) = file_list.item(i) {
                if file.type_().starts_with("image/") {
                    files_to_process.push(GlooFile::from(file));
                } else {
                    log::warn!("Skipping non-image file: {}", file.name());
                    ctx.link().send_message(
                        Msg::SetError(Some(format!("Skipped non-image file: {}", file.name())))
                    );
                }
            }
        }

        if !files_to_process.is_empty() {
            ctx.link().send_message(Msg::FilesAdded(files_to_process));
        }
    }

    fn send_analysis_request(&self, ctx: &Context<Self>, file_id: u64, file: GlooFile) {
        spawn_local({
            let link = ctx.link().clone();

            async move {
                let form_data = web_sys::FormData::new().unwrap();
                form_data.append_with_blob("image", file.as_ref()).unwrap();

                let request = Request::post("/api/inference")
                    .body(form_data)
                    .expect("Failed to build request.");

                match request.send().await {
                    Ok(response) => {
                        if response.ok() {
                            match response.json::<InferenceResponse>().await {
                                Ok(results) => {
                                    link.send_message(Msg::InferenceResult(file_id, results))
                                }
                                Err(e) =>
                                    link.send_message(
                                        Msg::SetError(
                                            Some(format!("Failed to parse response: {}", e))
                                        )
                                    ),
                            }
                        } else {
                            let status = response.status();
                            let body = response.text().await.unwrap_or_default();
                            link.send_message(
                                Msg::SetError(Some(format!("Server error: {} - {}", status, body)))
                            )
                        }
                    }
                    Err(e) => {
                        link.send_message(Msg::SetError(Some(format!("Network error: {}", e))))
                    }
                }
            }
        });
    }
}

// Rendering methods
impl Model {
    fn render_header(&self) -> Html {
        html! {
            <header class="app-header">
                <h1><i class="fa-solid fa-images"></i> {" Multi-Image Upload & Analysis"}</h1>
                <p class="subtitle">{"Upload images via button, drag & drop, or paste"}</p>
            </header>
        }
    }

    fn render_theme_toggle(&self, ctx: &Context<Self>) -> Html {
        let link = ctx.link();

        html! {
            <div class="top-right">
                <button
                    id="theme-toggle"
                    class="theme-toggle"
                    onclick={link.callback(|_| Msg::ToggleTheme)}
                    title={ if self.theme == "light" { "Switch to Dark Mode" } else { "Switch to Light Mode" } }
                >
                    { if self.theme == "light" {
                        html! { <img src="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/2600.svg" alt="Sun Icon" class="toggle-icon" /> }
                    } else {
                        html! { <img src="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f319.svg" alt="Moon Icon" class="toggle-icon" /> }
                    }}
                </button>
            </div>
        }
    }

    fn render_upload_section(&self, ctx: &Context<Self>) -> Html {
        let limit_reached = self.files.len() >= 15;
        html! {
            <div class="upload-section">
                { self.render_file_input_area(ctx, limit_reached) }
                { self.render_preview_area(ctx) }
            </div>
        }
    }

    fn extract_image_files(file_list: web_sys::FileList) -> Vec<GlooFile> {
        (0..file_list.length())
            .filter_map(|i| file_list.item(i))
            .filter(|file| file.type_().starts_with("image/"))
            .map(GlooFile::from)
            .collect()
    }

    fn render_file_input_area(&self, ctx: &Context<Self>, limit_reached: bool) -> Html {
        if limit_reached {
            return html! {
                <p class="limit-reached">{"You have reached the maximum of 15 images."}</p>
            };
        }

        let link = ctx.link();
        let handle_change = link.callback(|e: Event| {
            let input: HtmlInputElement = e.target_unchecked_into();
            if let Some(files) = input.files() {
                let files_to_process = Self::extract_image_files(files);
                Msg::FilesAdded(files_to_process)
            } else {
                Msg::SetError(Some("Failed to get files from input".into()))
            }
        });

        let handle_drag_over = link.callback(|e: DragEvent| {
            e.prevent_default();
            Msg::SetDragging(true)
        });

        let handle_drag_leave = link.callback(|e: DragEvent| {
            e.prevent_default();
            Msg::SetDragging(false)
        });

        let handle_drop = link.callback(Msg::HandleDrop);
        let trigger_file_input = Callback::from(|_| {
            if
                let Some(input) = web_sys
                    ::window()
                    .unwrap()
                    .document()
                    .unwrap()
                    .get_element_by_id("file-input")
            {
                if let Ok(html_input) = input.dyn_into::<web_sys::HtmlElement>() {
                    html_input.click();
                }
            }
        });

        html! {
            <>
                <input
                    type="file"
                    id="file-input"
                    multiple=true
                    accept="image/*"
                    style="display: none;"
                    onchange={handle_change}
                />

                <button
                    id="upload-button"
                    class="analyze-btn"
                    onclick={trigger_file_input.clone()}
                >
                    <i class="fa-solid fa-upload"></i> {" Select Images"}
                </button>

                <div
                    id="drop-zone"
                    class={classes!("upload-area", self.is_dragging.then_some("drag-over"))}
                    ondragover={handle_drag_over}
                    ondragleave={handle_drag_leave}
                    ondrop={handle_drop}
                    onclick={trigger_file_input}
                >
                    <div class="upload-placeholder">
                        <i class="fa-solid fa-cloud-arrow-up"></i>
                        <p>{"Drag & drop images here, paste, or click"}</p>
                        <p class="file-types">{"Supported formats: JPG, PNG, WEBP, GIF"}</p>
                    </div>
                </div>
            </>
        }
    }

    fn render_preview_area(&self, ctx: &Context<Self>) -> Html {
        if self.files.is_empty() {
            return html! {};
        }

        let link = ctx.link();

        html! {
            <div id="preview-container">
                { self.render_selected_image_preview() }
                <h2>{ format!("Previews: {} / 15", self.files.len()) }</h2>
                <div id="image-previews">
                    {{
                        let mut sorted_files: Vec<&FileData> = self.files.values().collect();
                        sorted_files.sort_by_key(|fd| fd.id);
                        sorted_files.iter()
                            .map(|file_data| self.render_preview_item(ctx, file_data))
                            .collect::<Html>()
                    }}
                </div>
                <div class="button-container">
                    <button
                        class="analyze-btn"
                        style="background-color: var(--danger-color);"
                        onclick={link.callback(|_| Msg::ClearAllFiles)}
                    >
                        <i class="fa-solid fa-trash"></i>{" Clear All"}
                    </button>
                    <button
                        class="analyze-btn"
                        onclick={link.callback(|_| Msg::AnalyzeSelected)}
                        disabled={self.loading || self.selected_file_id.is_none()}
                    >
                        { self.render_analyze_button_content() }
                    </button>
                    <button
                        class="analyze-btn"
                        style="background-color: var(--primary-color);"
                        onclick={link.callback(|_| Msg::AnalyzeAll)}
                    >
                        <i class="fa-solid fa-magnifying-glass"></i>{" Analyze All"}
                    </button>
                </div>
            </div>
        }
    }

    fn render_selected_image_preview(&self) -> Html {
        if
            let Some(url) = self.selected_file_id
                .and_then(|id| self.files.get(&id))
                .and_then(|file_data| file_data.preview_url.as_ref())
        {
            html! {
                <img id="actual-image-preview"
                    src={url.to_string()}
                    alt="Image Preview"
                    style="max-width:100%; max-height: 400px; object-fit: contain; margin-bottom: 10px;" />
            }
        } else {
            html! {}
        }
    }

    fn render_analyze_button_content(&self) -> Html {
        if self.loading {
            html! { <><i class="fa-solid fa-spinner fa-spin"></i>{" Analyzing..."}</> }
        } else {
            let filename = self.selected_file_id
                .and_then(|id| self.files.get(&id))
                .map(|fd| fd.file.name())
                .unwrap_or_else(|| "Selected Image".to_string());

            let display_name = if filename.len() > 20 {
                format!("{}...", &filename[..17])
            } else {
                filename
            };

            html! { <><i class="fa-solid fa-magnifying-glass"></i>{ format!(" Analyze \"{}\"", display_name) }</> }
        }
    }

    fn render_preview_item(&self, ctx: &Context<Self>, file_data: &FileData) -> Html {
        let file_id = file_data.id;
        let link = ctx.link();
        let is_selected = self.selected_file_id == Some(file_id);

        html! {
            <div
                class={classes!("preview-item", is_selected.then_some("selected"))}
                key={file_id.to_string()}
                onclick={link.callback(move |_| Msg::SelectFile(file_id))}
                style={if is_selected { "border: 2px solid var(--primary-color); box-shadow: 0 0 8px var(--primary-color);" } else { "" }}
                title={format!("Click to select for analysis: {}", file_data.file.name())}
            >
                {
                    if let Some(url) = &file_data.preview_url {
                        html! { <img src={url.to_string()} alt={file_data.file.name()} /> }
                    } else {
                        html! { <div class="preview-placeholder" style="height: 120px; display: flex; align-items: center; justify-content: center;">{"..."}</div> }
                    }
                }
                <button
                    class="remove-btn"
                    title="Remove this image"
                    onclick={link.callback(move |e: MouseEvent| {
                        e.stop_propagation();
                        Msg::RemoveFile(file_id)
                    })}
                >
                    <i class="fa-solid fa-times" style="font-size: 10px;"></i>
                </button>
            </div>
        }
    }

    fn render_error_message(&self) -> Html {
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

    fn render_results(&self) -> Html {
        if let Some(selected_id) = self.selected_file_id {
            if let Some(results) = self.results.get(&selected_id) {
                let predicted_class = results.predictions
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let confidence = results.predictions[predicted_class] * 100.0;
                let is_ai = predicted_class == 0;
                let analyzed_filename = self.files
                    .get(&selected_id)
                    .map(|fd| fd.file.name())
                    .unwrap_or_else(|| "Analyzed Image".to_string());

                html! {
                    <div class={classes!("results-container", if is_ai { "ai-detected" } else { "not-ai" })}>
                        <div class="result-header">
                            <h2 title={format!("Analysis results for: {}", analyzed_filename)}>
                                {
                                    if is_ai {
                                        html! { <><i class="fa-solid fa-robot"></i>{" AI Generated"}</> }
                                    } else {
                                        html! { <><i class="fa-solid fa-camera"></i>{" Likely Authentic"}</> }
                                    }
                                }
                                <span class="analyzed-filename-display">{format!("({})", analyzed_filename)}</span>
                            </h2>
                            <div class="confidence-meter">
                                <div class="meter-label">{"Confidence:"}</div>
                                <div class="meter">
                                    <div class="meter-fill" style={format!("width: {}%", confidence)}></div>
                                </div>
                                <div class="meter-value">{format!("{:.1}%", confidence)}</div>
                            </div>
                        </div>
                        <div class="detailed-results">
                            <h3>{"Detailed Analysis"}</h3>
                            <div class="result-bars">
                                { for results.predictions.iter().enumerate().map(|(i, &v)| {
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
                html! { <p>{"No analysis result available for the selected image."}</p> }
            }
        } else {
            html! {}
        }
    }
}
fn main() {
    wasm_logger::init(wasm_logger::Config::default());
    log::info!("App starting...");
    yew::Renderer::<Model>::new().render();
}
