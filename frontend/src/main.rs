use gloo_events::EventListener;
use gloo_file::{ File as GlooFile, ObjectUrl };
use shared::InferenceResponse;
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use web_sys::ClipboardEvent;
use yew::prelude::*;
use gloo_timers::callback::Timeout;

mod components;
mod api;
use components::header::render_header;
use components::theme_toggle::render_theme_toggle;
use components::upload_section::render_upload_section;
use components::preview_area::render_preview_area;
use components::results::render_results;
use components::handlers::*;
use components::utils::render_error_message;
use api::TaskStatus;

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
    pub task_map: HashMap<u64, String>,
}

#[derive(Clone)]
pub struct FileData {
    pub id: u64,
    pub file: GlooFile,
    pub preview_url: Option<ObjectUrl>,
}
pub enum Msg {
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
    PreviewLoaded,
    SetTaskMap(HashMap<u64, String>),

    // Input events
    HandleDrop(DragEvent),
    HandlePaste(ClipboardEvent),
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
            future_requests: 0,
            preview_loading: false,
            preview_load_timeout: None,
            task_map: HashMap::new(),
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
            Msg::ClearAllFiles => handle_clear_all_files(self),

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
            Msg::SetTaskMap(task_map) => {
                self.task_map = task_map;
                true
            }

            Msg::HandleDrop(event) => handle_drop(self, ctx, event),
            Msg::HandlePaste(event) => handle_paste(self, ctx, event),
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div class="container">
                { render_header() }
                { render_theme_toggle(&self.theme, ctx.link()) }

                <main class="main-content">
                { render_upload_section(self, ctx) }
                { render_preview_area(self, ctx) }
                { render_error_message(self) }
                { render_results(self) }
                {
                    if let Some(file_id) = self.selected_file_id {
                        let task_id_opt = self.task_map.get(&file_id);
                        if let Some(task_id) = task_id_opt {
                            html! { <TaskStatus task_id={Some(task_id.clone())} /> }
                        } else {
                            html! { <TaskStatus task_id={Option::<String>::None} /> }
                        }
                    } else {
                        html! { <TaskStatus task_id={Option::<String>::None} /> }
                    }
                }
                </main>

                <footer class="app-footer">
                    <p>{"Multi-Image Upload Demo | Fullstack Rust WASM"}</p>
                </footer>
            </div>
        }
    }
}

fn main() {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Info));
    log::info!("App starting...");
    yew::Renderer::<Model>::new().render();
}
