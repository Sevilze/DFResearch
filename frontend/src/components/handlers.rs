use yew::prelude::*;
use super::super::{Model, FileData};
use super::super::Msg;
use crate::components::utils::generate_id;
use gloo_file::{File as GlooFile, ObjectUrl};
use shared::InferenceResponse;
use gloo_timers::callback::Timeout;
use wasm_bindgen_futures::spawn_local;
use web_sys::{DragEvent, ClipboardEvent, FileList};
use gloo_net::http::Request;
use log;

pub fn handle_files_added(model: &mut Model, ctx: &Context<Model>, files: Vec<GlooFile>) -> bool {
    let current_count = model.files.len();
    let available_slots = (15usize).saturating_sub(current_count);

    if files.len() > available_slots {
        model.error = Some(
            format!("Upload limit exceeded. You can only add {} more images.", available_slots)
        );
        return true;
    }

    model.error = None;
    let mut new_selection = None;

    for file in files.into_iter() {
        let id = generate_id();
        let file_data = FileData {
            id,
            file: file.clone(),
            preview_url: None,
        };
        model.files.insert(id, file_data);

        let preview_url = ObjectUrl::from(file);
        ctx.link().send_message(Msg::AddPreview(id, preview_url));

        new_selection = Some(id);
    }

    if let Some(id) = new_selection {
        model.selected_file_id = Some(id);
    }

    true
}

pub fn handle_add_preview(model: &mut Model, id: u64, url: ObjectUrl) -> bool {
    if let Some(file_data) = model.files.get_mut(&id) {
        file_data.preview_url = Some(url);
        true
    } else {
        false
    }
}

pub fn handle_remove_file(model: &mut Model, id: u64) -> bool {
    if let Some(removed_file) = model.files.remove(&id) {
        drop(removed_file);
        model.results.remove(&id);

        if model.selected_file_id == Some(id) {
            model.selected_file_id = None;
        }

        if model.files.is_empty() {
            model.selected_file_id = None;
            model.results.clear();
            model.error = None;
        } else if model.selected_file_id.is_none() {
            model.selected_file_id = model.files.keys().last().cloned();
        }

        true
    } else {
        false
    }
}

pub fn handle_analyze_all(model: &mut Model, ctx: &Context<Model>) -> bool {
    model.loading = true;
    model.error = None;
    model.future_requests = model.files.len();

    for file_data in model.files.values().cloned().collect::<Vec<_>>() {
        let file = file_data.file.clone();
        let file_id = file_data.id;
        send_analysis_request(ctx, file_id, file);
    }

    true
}

pub fn handle_inference_result(model: &mut Model, file_id: u64, response: InferenceResponse) -> bool {
    model.results.insert(file_id, response);
    model.future_requests -= 1;
    if model.future_requests == 0 {
        model.loading = false;
    }
    true
}

pub fn handle_preview_loaded(model: &mut Model) -> bool {
    model.preview_loading = false;
    model.preview_load_timeout = None;
    true
}

pub fn handle_toggle_theme(model: &mut Model) -> bool {
    let body = web_sys::window().unwrap().document().unwrap().body().unwrap();

    if model.theme == "light" {
        model.theme = "dark".to_string();
        body.class_list().add_1("dark-mode").unwrap();
    } else {
        model.theme = "light".to_string();
        body.class_list().remove_1("dark-mode").unwrap();
    }

    true
}

pub fn handle_drop(model: &mut Model, ctx: &Context<Model>, event: DragEvent) -> bool {
    event.prevent_default();
    model.is_dragging = false;

    if let Some(data_transfer) = event.data_transfer() {
        if let Some(file_list) = data_transfer.files() {
            process_file_list(ctx, file_list);
        }
    }

    true
}

pub fn handle_paste(_model: &mut Model, ctx: &Context<Model>, event: ClipboardEvent) -> bool {
    if let Some(data_transfer) = event.clipboard_data() {
        if let Some(file_list) = data_transfer.files() {
            event.prevent_default();
            process_file_list(ctx, file_list);
            return true;
        }
    }
    false
}

pub fn process_file_list(ctx: &Context<Model>, file_list: FileList) {
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

pub fn send_analysis_request(ctx: &Context<Model>, file_id: u64, file: GlooFile) {
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

pub fn handle_select_file(model: &mut Model, ctx: &Context<Model>, id: u64) -> bool {
    if model.selected_file_id != Some(id) && model.files.contains_key(&id) {
        if let Some(timeout) = model.preview_load_timeout.take() {
            timeout.cancel();
        }

        model.selected_file_id = Some(id);
        model.error = None;
        model.preview_loading = true;

        let link = ctx.link().clone();
        let timeout = Timeout::new(0, move || {
            link.send_message(Msg::PreviewLoaded);
        });
        model.preview_load_timeout = Some(timeout);

        true
    } else {
        false
    }
}

pub fn handle_clear_all_files(model: &mut Model) -> bool {
    for (_, file_data) in model.files.iter_mut() {
        let _ = file_data.preview_url.take();
    }
    model.files.clear();
    model.selected_file_id = None;
    model.results.clear();
    model.error = None;
    true
}

pub fn handle_analyze_selected(model: &mut Model, ctx: &Context<Model>) -> bool {
    if let Some(file_id) = model.selected_file_id {
        if let Some(file_data) = model.files.get(&file_id) {
            model.loading = true;
            model.error = None;
            model.future_requests = 1;
            let file = file_data.file.clone();

            send_analysis_request(ctx, file_id, file);
            return true;
        }
    }

    ctx.link().send_message(Msg::SetError(Some("No file selected for analysis.".into())));
    false
}
