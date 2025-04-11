use yew::prelude::*;
use gloo_file::{File as GlooFile, ObjectUrl};
use shared::InferenceResponse;
use gloo_timers::callback::Timeout;
use web_sys::{DragEvent, ClipboardEvent, FileList};
use gloo_net::http::Request;
use wasm_bindgen::JsCast;
use super::super::{Model, FileData, Msg};
use super::utils::generate_id;

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

pub fn send_inference_request(ctx: &Context<Model>, files: Vec<FileData>) {
    let link = ctx.link().clone();

    wasm_bindgen_futures::spawn_local(async move {
        let form_data = web_sys::FormData::new().unwrap();

        for file_data in &files {
            form_data.append_with_blob("image", file_data.file.as_ref()).unwrap();
        }

        let request = Request::post("/api/inference")
            .body(form_data)
            .expect("Failed to build request.");

        match request.send().await {
            Ok(response) => {
                if response.ok() {
                    let text = response.text().await.unwrap_or_default();
                    match serde_json::from_str::<serde_json::Value>(&text) {
                        Ok(json_value) => {
                            if let Some(results_array) = json_value.get("results").and_then(|v| v.as_array()) {
                                let mut task_map = std::collections::HashMap::new();
                                for (file_data, result_value) in files.iter().zip(results_array) {
                                    if let Some(task_obj) = result_value.get("task") {
                                        if let Some(task_id) = task_obj.get("id").and_then(|id| id.as_str()) {
                                            task_map.insert(file_data.id, task_id.to_string());
                                        }
                                    }
                                    if let Some(inf) = result_value.get("inference") {
                                        match serde_json::from_value::<InferenceResponse>(inf.clone()) {
                                            Ok(results) => {
                                                link.send_message(Msg::InferenceResult(file_data.id, results))
                                            }
                                            Err(e) => link.send_message(
                                                Msg::SetError(Some(format!("Failed to parse inference: {}", e)))
                                            ),
                                        }
                                    }
                                }
                                link.send_message(Msg::SetTaskMap(task_map));
                            } else {
                                link.send_message(
                                    Msg::SetError(Some("No results array in response.".to_string()))
                                );
                            }
                        }
                        Err(e) => link.send_message(
                            Msg::SetError(Some(format!("Failed to parse response: {}", e)))
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
    });
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

pub fn handle_clear_all_files(_model: &mut Model, ctx: &Context<Model>) -> bool {
    let window = match web_sys::window() {
        Some(win) => win,
        None => {
            ctx.link().send_message(Msg::InternalExecuteClearAll);
            return false;
        }
    };
    let document = match window.document() {
        Some(doc) => doc,
        None => {
            ctx.link().send_message(Msg::InternalExecuteClearAll);
            return false;
        }
    };
    let body = match document.body() {
        Some(b) => b,
        None => {
            ctx.link().send_message(Msg::InternalExecuteClearAll);
            return false;
        }
    };

    let button_element = document.get_element_by_id("clear-all-btn").and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok());
    let (button_x, button_y) = match button_element {
        Some(btn) => {
            let rect = btn.get_bounding_client_rect();
            (rect.left() + rect.width() / 2.0, rect.top() + rect.height() / 2.0)
        }
        None => {
            let win_width = window.inner_width().unwrap().as_f64().unwrap_or(0.0);
            let win_height = window.inner_height().unwrap().as_f64().unwrap_or(0.0);
            (win_width / 2.0, win_height / 2.0)
        }
    };

    let circle = document.create_element("div").unwrap().dyn_into::<web_sys::HtmlElement>().unwrap();
    circle.class_list().add_1("xray-circle").unwrap();

    let win_width = window.inner_width().unwrap().as_f64().unwrap_or(1000.0);
    let win_height = window.inner_height().unwrap().as_f64().unwrap_or(1000.0);
    let max_dim = win_width.max(win_height);
    let circle_diameter = max_dim * 2.0;

    let style = format!(
        "position: fixed; top: {}px; left: {}px; width: {}px; height: {}px; transform: translate(-50%, -50%) scale(0); border-radius: 50%; z-index: 9999; pointer-events: none;",
        button_y, button_x, circle_diameter, circle_diameter
    );
    circle.set_attribute("style", &style).unwrap();

    body.append_child(&circle).unwrap();
    let _ = circle.offset_width();
    circle.class_list().add_1("expanding").unwrap();

    let link = ctx.link().clone();
    let circle_clone_phase2 = circle.clone();
    let body_clone = body.clone();

    body.class_list().add_1("xray-active").unwrap();

    gloo_timers::callback::Timeout::new(500, move || {
        link.send_message(Msg::InternalExecuteClearAll);
        let circle_clone_phase3 = circle_clone_phase2.clone();
        let body_clone2 = body_clone.clone();
        circle_clone_phase3.class_list().remove_1("expanding").unwrap();
        circle_clone_phase3.class_list().add_1("contracting").unwrap();
        let _ = body_clone2.class_list().remove_1("xray-active");
    }).forget();

    let circle_clone_cleanup = circle.clone();
    gloo_timers::callback::Timeout::new(2000, move || {
        if let Some(parent) = circle_clone_cleanup.parent_node() {
            let _ = parent.remove_child(&circle_clone_cleanup);
        }
    }).forget();

    false
}

pub fn handle_internal_execute_clear_all(model: &mut Model) -> bool {
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

            let files = vec![file_data.clone()];
            send_inference_request(ctx, files);
            return true;
        }
    }

    ctx.link().send_message(Msg::SetError(Some("No file selected for analysis.".into())));
    false
}

pub fn handle_analyze_all(model: &mut Model, ctx: &Context<Model>) -> bool {
    model.loading = true;
    model.error = None;
    model.future_requests = model.files.len();

    let files: Vec<_> = model.files.values().cloned().collect();
    send_inference_request(ctx, files);

    true
}
