use gloo_events::EventListener;
use gloo_file::{File as GlooFile, ObjectUrl};
use gloo_storage::{LocalStorage, Storage};
use gloo_timers::callback::Timeout;
use shared::{InferenceResponse, ProcessingMode};
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use web_sys::{ClipboardEvent, UrlSearchParams};
use yew::prelude::*;

mod api;
mod components;

use components::auth_button::AuthButton;
use components::handlers::*;
use components::header::render_header;
use components::preview_area::render_preview_area;
use components::results::render_results;
use components::theme_toggle::render_theme_toggle;
use components::upload_section::render_upload_section;
use components::utils::render_error_message;

pub struct Model {
    pub files: HashMap<u64, FileData>,
    pub selected_file_id: Option<u64>,
    pub results: HashMap<u64, InferenceResponse>,
    pub loading: bool,
    pub error: Option<String>,
    pub is_dragging: bool,
    pub paste_listener: Option<EventListener>,
    pub theme: String,
    pub future_requests: usize,
    pub preview_loading: bool,
    pub preview_load_timeout: Option<Timeout>,

    pub processing_mode: ProcessingMode,
    pub auth_token: Option<String>,
    pub completed_requests: usize,
    pub batch_processing: bool,
    pub inference_status: HashMap<u64, InferenceStatus>,
}

#[derive(Clone, PartialEq)]
pub enum InferenceStatus {
    None,
    Processing,
    Completed,
}

#[derive(Clone)]
pub struct FileData {
    pub id: u64,
    pub file: GlooFile,
    pub preview_url: Option<ObjectUrl>,
    pub image_hash: Option<String>,
    pub is_cached: bool,
}

pub enum Msg {
    FilesAdded(Vec<GlooFile>),
    AddPreview(u64, ObjectUrl),
    RemoveFile(u64),
    SelectFile(u64),
    ClearAllFiles,
    InternalExecuteClearAll,
    AnalyzeSelected,
    AnalyzeAll,
    InferenceResult(u64, InferenceResponse),
    SetError(Option<String>),
    SetDragging(bool),
    ToggleTheme,
    PreviewLoaded,
    RestoreCachedFile(u64, FileData),
    HandleDrop(DragEvent),
    HandlePaste(ClipboardEvent),
    SetProcessingMode(ProcessingMode),
    ResetLoadingState,
    SetFutureRequests(usize),
    SetAuthToken(Option<String>),
    IncrementCompletedRequests,
    StartBatchProcessing(usize),
    SetInferenceStatus(u64, InferenceStatus),
}

impl Model {
    fn extract_token_from_url(ctx: &Context<Self>) {
        let window = match web_sys::window() {
            Some(w) => w,
            None => return,
        };
        let location = window.location();
        let search = match location.search() {
            Ok(s) => s,
            Err(_) => return,
        };

        if search.is_empty() {
            let stored_token: Option<String> = LocalStorage::get("auth_token").ok();
            if let Some(ref _token) = stored_token {
                // Update component state with stored token
                ctx.link()
                    .send_message(Msg::SetAuthToken(stored_token.clone()));
            }
            return;
        }

        let url_params = match UrlSearchParams::new_with_str(&search) {
            Ok(params) => params,
            Err(_) => return,
        };
        let token = url_params.get("token");

        if let Some(ref token_str) = token {
            match LocalStorage::set("auth_token", token_str) {
                Ok(_) => (),
                Err(e) => log::error!("Failed to store token in localStorage: {:?}", e),
            }

            let new_url = format!(
                "{}{}",
                location.origin().unwrap_or_default(),
                location.pathname().unwrap_or_default()
            );
            let _ = window.history().unwrap().replace_state_with_url(
                &wasm_bindgen::JsValue::NULL,
                "",
                Some(&new_url),
            );

            // Update component state with new token
            ctx.link().send_message(Msg::SetAuthToken(token.clone()));
        } else {
            let stored_token: Option<String> = LocalStorage::get("auth_token").ok();
            if stored_token.is_some() {
                // Update component state with stored token
                ctx.link()
                    .send_message(Msg::SetAuthToken(stored_token.clone()));
            }
        }
    }
}

impl Component for Model {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        Self::extract_token_from_url(ctx);

        let mut model = Self {
            files: HashMap::new(),
            selected_file_id: None,
            results: HashMap::new(),
            loading: false,
            error: None,
            is_dragging: false,
            paste_listener: None,
            theme: "light".to_string(),
            future_requests: 0,
            preview_loading: false,
            preview_load_timeout: None,

            processing_mode: ProcessingMode::IntermediateFusionEnsemble,
            auth_token: None,
            completed_requests: 0,
            batch_processing: false,
            inference_status: HashMap::new(),
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
            Msg::FilesAdded(files) => handle_files_added(self, ctx, files),
            Msg::AddPreview(id, url) => handle_add_preview(self, id, url),
            Msg::RemoveFile(id) => handle_remove_file(self, id),
            Msg::SelectFile(id) => handle_select_file(self, ctx, id),
            Msg::ClearAllFiles => handle_clear_all_files(self, ctx),
            Msg::AnalyzeSelected => handle_analyze_selected(self, ctx),
            Msg::AnalyzeAll => handle_analyze_all(self, ctx),
            Msg::InferenceResult(file_id, response) => {
                handle_inference_result(self, file_id, response)
            }
            Msg::SetError(error) => {
                self.error = error;
                self.loading = false;
                true
            }
            Msg::SetDragging(is_dragging) => {
                self.is_dragging = is_dragging;
                true
            }
            Msg::ToggleTheme => handle_toggle_theme(self),
            Msg::PreviewLoaded => handle_preview_loaded(self),
            Msg::RestoreCachedFile(file_id, file_data) => {
                self.files.insert(file_id, file_data);
                if self.selected_file_id.is_none() {
                    self.selected_file_id = Some(file_id);
                }
                true
            }
            Msg::HandleDrop(event) => handle_drop(self, ctx, event),
            Msg::HandlePaste(event) => handle_paste(self, ctx, event),
            Msg::InternalExecuteClearAll => handle_internal_execute_clear_all(self),
            Msg::SetProcessingMode(mode) => handle_set_processing_mode(self, ctx, mode),
            Msg::ResetLoadingState => {
                self.loading = false;
                self.future_requests = 0;
                true
            }
            Msg::SetFutureRequests(count) => {
                self.future_requests = count;
                self.loading = count > 0;
                true
            }
            Msg::SetAuthToken(token) => {
                let had_no_token = self.auth_token.is_none();
                let now_has_token = token.is_some();
                self.auth_token = token;

                if now_has_token && (had_no_token || self.files.is_empty()) {
                    fetch_and_restore_session(ctx, self.processing_mode.clone());
                }

                true
            }
            Msg::IncrementCompletedRequests => {
                self.completed_requests += 1;
                if self.completed_requests >= self.future_requests {
                    self.loading = false;
                    self.batch_processing = false;
                    self.completed_requests = 0;
                }
                true
            }
            Msg::StartBatchProcessing(total) => {
                self.batch_processing = true;
                self.completed_requests = 0;
                self.future_requests = total;
                self.loading = true;
                true
            }
            Msg::SetInferenceStatus(file_id, status) => {
                self.inference_status.insert(file_id, status);
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let on_token_change = {
            let link = ctx.link().clone();
            Callback::from(move |token: Option<String>| {
                link.send_message(Msg::SetAuthToken(token));
            })
        };

        html! {
            <div class="container">
                <div class="top-controls">
                    <AuthButton
                        auth_token={self.auth_token.clone()}
                        on_token_change={on_token_change}
                    />
                    { render_theme_toggle(&self.theme, ctx.link()) }
                </div>
                { render_header() }
                <main class="main-content">
                    { render_upload_section(self, ctx) }
                    { render_preview_area(self, ctx) }
                    { render_error_message(self) }
                    { render_results(self) }

                </main>
                <footer class="app-footer">
                    <p>{"Fullstack Rust WASM"}</p>
                </footer>
            </div>
        }
    }
}

fn main() {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Info));
    yew::Renderer::<Model>::new().render();
}
