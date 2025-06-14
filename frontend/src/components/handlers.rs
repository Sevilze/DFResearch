use super::super::{FileData, InferenceStatus, Model, Msg};
use super::utils::generate_id;
use gloo_file::{File as GlooFile, ObjectUrl};
use gloo_net::http::{Request, RequestBuilder};
use gloo_storage::{LocalStorage, Storage};
use gloo_timers::callback::Timeout;
use serde_json::Value;
use shared::{InferenceResponse, ProcessingMode};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::{spawn_local, JsFuture};
use web_sys::{
    ClipboardEvent, DragEvent, FileList, FormData, ReadableStreamDefaultReader, TextDecoder,
};
use yew::prelude::*;

pub fn handle_files_added(model: &mut Model, ctx: &Context<Model>, files: Vec<GlooFile>) -> bool {
    let current_count = model.files.len();
    let available_slots = (15usize).saturating_sub(current_count);

    if files.len() > available_slots {
        model.error = Some(format!(
            "Upload limit exceeded. You can only add {} more images.",
            available_slots
        ));
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
            image_hash: None,
            is_cached: false,
        };
        model.files.insert(id, file_data);

        let preview_url = ObjectUrl::from(file);
        ctx.link().send_message(Msg::AddPreview(id, preview_url));
        if new_selection.is_none() {
            new_selection = Some(id);
        }
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
        // If this is a cached file, delete it from the backend
        if removed_file.is_cached {
            if let Some(image_hash) = &removed_file.image_hash {
                let image_hash = image_hash.clone();
                spawn_local(async move {
                    if let Err(e) = delete_cached_image(&image_hash).await {
                        log::error!("Failed to delete cached image {}: {:?}", image_hash, e);
                    }
                });
            }
        }

        drop(removed_file);
        model.results.remove(&id);
        model.inference_status.remove(&id);

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

fn get_auth_token() -> Option<String> {
    LocalStorage::get("auth_token").ok()
}

// Build an authenticated request builder with the given method and endpoint
fn build_authenticated_request(method: &str, endpoint: &str) -> Option<RequestBuilder> {
    let auth_token = get_auth_token()?;

    let request_builder = match method {
        "GET" => Request::get(endpoint),
        "POST" => Request::post(endpoint),
        "DELETE" => Request::delete(endpoint),
        _ => return None,
    };

    Some(request_builder.header("Authorization", &format!("Bearer {}", auth_token)))
}

pub fn inference_request_wrapper(ctx: &Context<Model>, files: Vec<FileData>, mode: ProcessingMode) {
    let link = ctx.link().clone();

    wasm_bindgen_futures::spawn_local(async move {
        let link_clone = link.clone();
        let send_error = move |msg: String| {
            link_clone.send_message(Msg::SetError(Some(msg)));
            link_clone.send_message(Msg::ResetLoadingState);
        };

        send_inference_request_sse(link, files, mode, send_error).await;
    });
}

async fn send_inference_request_sse(
    link: yew::html::Scope<Model>,
    files: Vec<FileData>,
    mode: ProcessingMode,
    send_error: impl Fn(String) + 'static,
) {
    let auth_token = get_auth_token();

    match prepare_inference_request(&files, mode, &link) {
        Ok((form_data, file_id_map)) => {
            let endpoint = if auth_token.is_some() {
                "/api/inference"
            } else {
                "/public/inference"
            };

            handle_streaming_request(
                endpoint,
                form_data,
                auth_token,
                file_id_map,
                link,
                send_error,
            )
            .await;
        }
        Err(error_msg) => {
            send_error(error_msg);
        }
    }
}

// Prepare the form data and file ID mapping for inference request
fn prepare_inference_request(
    files: &[FileData],
    mode: ProcessingMode,
    link: &yew::html::Scope<Model>,
) -> Result<(FormData, std::collections::HashMap<u64, u64>), String> {
    let form_data = FormData::new().unwrap();
    let mut file_id_map = std::collections::HashMap::new();
    let mut file_counter = 1u64;

    for file_data in files {
        if file_data.is_cached {
            if let Some(image_hash) = &file_data.image_hash {
                form_data
                    .append_with_str("cached_image_hash", image_hash)
                    .unwrap();
                form_data
                    .append_with_str("cached_filename", &file_data.file.name())
                    .unwrap();
            } else {
                return Err(format!(
                    "Cached file missing image hash: {}",
                    file_data.file.name()
                ));
            }
        } else {
            let file_blob = file_data.file.as_ref();
            form_data
                .append_with_blob_and_filename("image", file_blob, &file_data.file.name())
                .unwrap();
        }

        file_id_map.insert(file_counter, file_data.id);
        link.send_message(Msg::SetInferenceStatus(
            file_data.id,
            InferenceStatus::Processing,
        ));
        file_counter += 1;
    }

    let mode_str = match mode {
        ProcessingMode::IntermediateFusionEnsemble => "intermediate",
        ProcessingMode::LateFusionEnsemble => "late_fusion",
    };
    form_data.append_with_str("mode", mode_str).unwrap();

    Ok((form_data, file_id_map))
}

async fn handle_streaming_request(
    endpoint: &str,
    form_data: FormData,
    auth_token: Option<String>,
    file_id_map: std::collections::HashMap<u64, u64>,
    link: yew::html::Scope<Model>,
    send_error: impl Fn(String) + 'static,
) {
    let request = match build_streaming_request(endpoint, form_data, auth_token) {
        Ok(req) => req,
        Err(e) => {
            send_error(e);
            return;
        }
    };

    let response = match request.send().await {
        Ok(resp) if resp.ok() => resp,
        Ok(resp) => {
            let error_text = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown server error".to_string());
            send_error(format!("Server error in streaming: {}", error_text));
            return;
        }
        Err(e) => {
            send_error(format!("Network error in streaming request: {}", e));
            return;
        }
    };

    let Some(body) = response.body() else {
        send_error("Response body is missing.".to_string());
        return;
    };

    process_streaming_response(body, file_id_map, link, send_error).await;
}

// Build the streaming request with proper headers
fn build_streaming_request(
    endpoint: &str,
    form_data: FormData,
    auth_token: Option<String>,
) -> Result<Request, String> {
    let mut request_builder = Request::post(endpoint);

    if let Some(token) = auth_token {
        request_builder = request_builder.header("Authorization", &format!("Bearer {}", token));
    }

    request_builder
        .body(form_data)
        .map_err(|e| format!("Failed to build streaming request: {}", e))
}

// Process the streaming response and handle SSE events
async fn process_streaming_response(
    body: web_sys::ReadableStream,
    file_id_map: std::collections::HashMap<u64, u64>,
    link: yew::html::Scope<Model>,
    send_error: impl Fn(String) + 'static,
) {
    let reader: ReadableStreamDefaultReader = body.get_reader().dyn_into().unwrap();
    let decoder = TextDecoder::new().unwrap();
    let mut buffer = String::new();

    loop {
        match JsFuture::from(reader.read()).await {
            Ok(result) => {
                let result_obj: js_sys::Object = result.dyn_into().unwrap();
                let done_val = js_sys::Reflect::get(&result_obj, &"done".into()).unwrap();
                if done_val.as_bool().unwrap_or(true) {
                    break;
                }

                let chunk_val = js_sys::Reflect::get(&result_obj, &"value".into()).unwrap();
                let chunk_uint8: js_sys::Uint8Array = chunk_val.dyn_into().unwrap();
                buffer.push_str(&decoder.decode_with_buffer_source(&chunk_uint8).unwrap());

                process_buffer_lines(&mut buffer, &file_id_map, &link, &send_error);
            }
            Err(e) => {
                send_error(format!("Error reading from stream: {:?}", e));
                break;
            }
        }
    }

    link.send_message(Msg::ResetLoadingState);
}

// Process complete lines in the buffer and handle SSE events
fn process_buffer_lines(
    buffer: &mut String,
    file_id_map: &std::collections::HashMap<u64, u64>,
    link: &yew::html::Scope<Model>,
    send_error: &impl Fn(String),
) {
    while let Some(newline_pos) = buffer.find('\n') {
        let line = buffer.drain(..=newline_pos).collect::<String>();
        let line = line.trim();

        if let Some(data) = line.strip_prefix("data: ") {
            if data == "{\"type\": \"complete\"}" {
                continue;
            }

            match serde_json::from_str::<Value>(data) {
                Ok(event_data) => {
                    handle_sse_event(event_data, file_id_map, link, send_error);
                }
                Err(e) => log::error!("Failed to parse SSE event data: {}", e),
            }
        }
    }
}

// Handle individual SSE events
fn handle_sse_event(
    event_data: Value,
    file_id_map: &std::collections::HashMap<u64, u64>,
    link: &yew::html::Scope<Model>,
    send_error: &impl Fn(String),
) {
    if let Some(error) = event_data.get("error") {
        send_error(format!("Streaming error: {}", error));
        return;
    }

    let Some(file_id_backend) = event_data.get("file_id").and_then(|v| v.as_u64()) else {
        return;
    };

    let Some(inference_data) = event_data.get("inference") else {
        return;
    };

    let Some(&frontend_file_id) = file_id_map.get(&file_id_backend) else {
        return;
    };

    match serde_json::from_value::<InferenceResponse>(inference_data.clone()) {
        Ok(response) => {
            link.send_message(Msg::InferenceResult(frontend_file_id, response));
        }
        Err(e) => {
            log::error!("Failed to parse streaming inference response: {}", e);
        }
    }
}

pub fn handle_inference_result(
    model: &mut Model,
    file_id: u64,
    response: InferenceResponse,
) -> bool {
    model.results.insert(file_id, response.clone());
    model
        .inference_status
        .insert(file_id, InferenceStatus::Completed);

    // Update the FileData to mark it as cached and set the image hash
    if let Some(file_data) = model.files.get_mut(&file_id) {
        if !file_data.is_cached && response.image_hash.is_some() {
            // This image was just processed and cached
            file_data.is_cached = true;
            file_data.image_hash = response.image_hash.clone();
        }
    }

    if model.batch_processing {
        model.completed_requests += 1;
        if model.completed_requests >= model.future_requests {
            model.loading = false;
            model.batch_processing = false;
            model.completed_requests = 0;
        }
    } else {
        model.future_requests = model.future_requests.saturating_sub(1);
        if model.future_requests == 0 {
            model.loading = false;
        }
    }

    true
}

pub fn handle_preview_loaded(model: &mut Model) -> bool {
    model.preview_loading = false;
    model.preview_load_timeout = None;
    true
}

pub fn handle_toggle_theme(model: &mut Model) -> bool {
    let body = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .body()
        .unwrap();

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
                ctx.link().send_message(Msg::SetError(Some(format!(
                    "Skipped non-image file: {}",
                    file.name()
                ))));
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
        Some(body) => body,
        None => {
            ctx.link().send_message(Msg::InternalExecuteClearAll);
            return false;
        }
    };

    let button_element = document
        .get_element_by_id("clear-all-btn")
        .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok());
    let (button_x, button_y) = match button_element {
        Some(btn) => {
            let rect = btn.get_bounding_client_rect();
            (
                rect.left() + rect.width() / 2.0,
                rect.top() + rect.height() / 2.0,
            )
        }
        None => {
            let win_width = window.inner_width().unwrap().as_f64().unwrap_or(0.0);
            let win_height = window.inner_height().unwrap().as_f64().unwrap_or(0.0);
            (win_width / 2.0, win_height / 2.0)
        }
    };

    let circle = document
        .create_element("div")
        .unwrap()
        .dyn_into::<web_sys::HtmlElement>()
        .unwrap();
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
        circle_clone_phase3
            .class_list()
            .remove_1("expanding")
            .unwrap();
        circle_clone_phase3
            .class_list()
            .add_1("contracting")
            .unwrap();
        let _ = body_clone2.class_list().remove_1("xray-active");
    })
    .forget();

    let circle_clone_cleanup = circle.clone();
    gloo_timers::callback::Timeout::new(2000, move || {
        if let Some(parent) = circle_clone_cleanup.parent_node() {
            let _ = parent.remove_child(&circle_clone_cleanup);
        }
    })
    .forget();

    false
}

pub fn handle_internal_execute_clear_all(model: &mut Model) -> bool {
    let has_cached_files = model.files.values().any(|file| file.is_cached);

    if has_cached_files {
        spawn_local(async move {
            if let Err(e) = clear_user_cache().await {
                log::error!("Failed to clear user cache: {:?}", e);
            }
        });
    }

    for (_, file_data) in model.files.iter_mut() {
        let _ = file_data.preview_url.take();
    }
    model.files.clear();
    model.selected_file_id = None;
    model.results.clear();
    model.inference_status.clear();
    model.error = None;
    true
}

pub fn handle_analyze_selected(model: &mut Model, ctx: &Context<Model>) -> bool {
    if let Some(file_id) = model.selected_file_id {
        if let Some(file_data) = model.files.get(&file_id) {
            model.results.remove(&file_id);
            // Only clear the status for the file being analyzed, not all files
            model.inference_status.remove(&file_id);
            model.loading = true;
            model.error = None;
            model.future_requests = 1;

            let files = vec![file_data.clone()];
            inference_request_wrapper(ctx, files, model.processing_mode.clone());
            return true;
        }
    }

    ctx.link()
        .send_message(Msg::SetError(Some("No file selected for analysis.".into())));
    false
}

pub fn handle_analyze_all(model: &mut Model, ctx: &Context<Model>) -> bool {
    model.results.clear();
    model.inference_status.clear();
    model.loading = true;
    model.error = None;
    model.future_requests = model.files.len();
    model.batch_processing = true;
    model.completed_requests = 0;

    let files: Vec<_> = model.files.values().cloned().collect();
    inference_request_wrapper(ctx, files, model.processing_mode.clone());

    true
}

pub fn handle_set_processing_mode(
    model: &mut Model,
    _ctx: &Context<Model>,
    mode: ProcessingMode,
) -> bool {
    if model.processing_mode != mode {
        model.processing_mode = mode;

        true
    } else {
        false
    }
}

pub fn fetch_and_restore_session(ctx: &Context<Model>, _current_processing_mode: ProcessingMode) {
    let link = ctx.link().clone();

    spawn_local(async move {
        if get_auth_token().is_none() {
            return;
        }

        match fetch_session_data().await {
            Ok(session_data) => {
                restore_cached_files(&link, &session_data).await;
                restore_inference_results(&link, &session_data);
            }
            Err(e) => {
                log::warn!("Failed to restore session: {}", e);
            }
        }
    });
}

// Fetch session data from the backend
async fn fetch_session_data() -> Result<Value, Box<dyn std::error::Error>> {
    let request_builder = build_authenticated_request("GET", "/api/cache/history")
        .ok_or("No authentication token available")?;

    let response = request_builder.send().await?;
    if !response.ok() {
        return Err(format!("Failed to fetch session data: {}", response.status()).into());
    }

    let session_data: Value = response.json().await?;
    Ok(session_data)
}

// Restore cached files from session data
async fn restore_cached_files(link: &yew::html::Scope<Model>, session_data: &Value) {
    let Some(images_obj) = session_data.get("images").and_then(|v| v.as_object()) else {
        return;
    };

    for (file_id_str, image_data) in images_obj {
        let Ok(file_id) = file_id_str.parse::<u64>() else {
            continue;
        };

        let Some(preview_url) = image_data.get("preview_url").and_then(|v| v.as_str()) else {
            continue;
        };

        let file_data = create_cached_file_data(file_id, image_data);
        link.send_message(Msg::RestoreCachedFile(file_id, file_data));

        // Fetch and set the preview URL asynchronously
        let link_clone = link.clone();
        let preview_url_owned = preview_url.to_string();
        spawn_local(async move {
            match fetch_cached_image_as_blob(&preview_url_owned).await {
                Ok(blob_url) => {
                    link_clone.send_message(Msg::AddPreview(file_id, blob_url));
                }
                Err(e) => {
                    log::error!(
                        "Failed to create blob URL for cached image {}: {:?}",
                        file_id,
                        e
                    );
                }
            }
        });
    }
}

// Create FileData for a cached image from session data
fn create_cached_file_data(file_id: u64, image_data: &Value) -> FileData {
    let file_name = image_data
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("cached_image");
    let file_size = image_data.get("size").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let image_hash = image_data
        .get("image_hash")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    FileData {
        id: file_id,
        file: create_mock_file(file_name, file_size),
        preview_url: None,
        image_hash,
        is_cached: true,
    }
}

// Restore inference results from session data
fn restore_inference_results(link: &yew::html::Scope<Model>, session_data: &Value) {
    let Some(results_obj) = session_data.get("results").and_then(|v| v.as_object()) else {
        return;
    };

    for (file_id_str, results_by_mode) in results_obj {
        let Ok(file_id) = file_id_str.parse::<u64>() else {
            continue;
        };

        let Some(results_map) = results_by_mode.as_object() else {
            continue;
        };

        // The backend will handle mode-specific caching during reanalysis
        if let Some((_, result_data)) = results_map.iter().next() {
            if let Ok(inference_response) =
                serde_json::from_value::<InferenceResponse>(result_data.clone())
            {
                link.send_message(Msg::InferenceResult(file_id, inference_response));
            }
        }
    }
}

// Create a mock file for cached images
fn create_mock_file(name: &str, size: u32) -> GlooFile {
    let array = js_sys::Uint8Array::new_with_length(size);
    let blob = web_sys::Blob::new_with_u8_array_sequence(&js_sys::Array::of1(&array)).unwrap();

    let file_options = web_sys::FilePropertyBag::new();
    file_options.set_type("image/jpeg");
    let file = web_sys::File::new_with_blob_sequence_and_options(
        &js_sys::Array::of1(&blob),
        name,
        &file_options,
    )
    .unwrap();

    GlooFile::from(file)
}

// Fetch cached image and create a blob URL
async fn fetch_cached_image_as_blob(
    preview_url: &str,
) -> Result<ObjectUrl, Box<dyn std::error::Error>> {
    let request_builder = build_authenticated_request("GET", preview_url)
        .ok_or("No authentication token available")?;

    let response = request_builder.send().await?;
    if !response.ok() {
        return Err(format!("Failed to fetch image: {}", response.status()).into());
    }

    let bytes = response.binary().await?;
    let uint8_array = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
    uint8_array.copy_from(&bytes);

    let blob = web_sys::Blob::new_with_u8_array_sequence(&js_sys::Array::of1(&uint8_array))
        .map_err(|e| format!("Failed to create blob: {:?}", e))?;

    Ok(ObjectUrl::from(blob))
}

async fn delete_cached_image(image_hash: &str) -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = format!("/api/cache/image/{}", image_hash);
    let request_builder = build_authenticated_request("DELETE", &endpoint)
        .ok_or("No authentication token available")?;

    let response = request_builder.send().await?;
    if !response.ok() {
        return Err(format!("Failed to delete cached image: {}", response.status()).into());
    }

    Ok(())
}

async fn clear_user_cache() -> Result<(), Box<dyn std::error::Error>> {
    let request_builder = build_authenticated_request("DELETE", "/api/cache/clear")
        .ok_or("No authentication token available")?;

    let response = request_builder.send().await?;
    if !response.ok() {
        return Err(format!("Failed to clear user cache: {}", response.status()).into());
    }

    Ok(())
}
