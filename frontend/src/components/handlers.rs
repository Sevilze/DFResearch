use super::super::{FileData, Model, Msg};
use super::utils::generate_id;
use gloo_file::{File as GlooFile, ObjectUrl};
use gloo_net::http::Request;
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

// No changes in this function
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

// No changes in this function
pub fn handle_add_preview(model: &mut Model, id: u64, url: ObjectUrl) -> bool {
    if let Some(file_data) = model.files.get_mut(&id) {
        file_data.preview_url = Some(url);
        true
    } else {
        false
    }
}

// No changes in this function
pub fn handle_remove_file(model: &mut Model, id: u64) -> bool {
    if let Some(removed_file) = model.files.remove(&id) {
        // If this is a cached file, delete it from the backend
        if removed_file.is_cached {
            if let Some(image_hash) = &removed_file.image_hash {
                let image_hash = image_hash.clone();
                spawn_local(async move {
                    if let Err(e) = delete_cached_image(&image_hash).await {
                        log::error!("Failed to delete cached image {}: {:?}", image_hash, e);
                    } else {
                        log::info!("Successfully deleted cached image: {}", image_hash);
                    }
                });
            }
        }

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

// No changes in this function
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

// No changes in this function
async fn send_inference_request_sse(
    link: yew::html::Scope<Model>,
    files: Vec<FileData>,
    mode: ProcessingMode,
    send_error: impl Fn(String) + 'static,
) {
    let auth_token: Option<String> = LocalStorage::get("auth_token").ok();
    let form_data = FormData::new().unwrap();

    // Add file IDs to track which file corresponds to which result
    let mut file_id_map = std::collections::HashMap::new();
    let mut file_counter = 1u64;

    for file_data in &files {
        let file_blob = file_data.file.as_ref();
        form_data
            .append_with_blob_and_filename("image", file_blob, &file_data.file.name())
            .unwrap();
        file_id_map.insert(file_counter, file_data.id);
        file_counter += 1;
    }

    let mode_str = match mode {
        ProcessingMode::IntermediateFusionEnsemble => "intermediate",
        ProcessingMode::LateFusionEnsemble => "late_fusion",
    };
    form_data.append_with_str("mode", mode_str).unwrap();

    // Determine which endpoint to use based on authentication status
    let endpoint = if auth_token.is_some() {
        log::info!("User authenticated - using authenticated streaming endpoint with caching");
        "/api/inference"
    } else {
        log::info!("User not authenticated - using public streaming endpoint (no caching)");
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

/// # Refactored `handle_streaming_request`
/// This function now implements true asynchronous stream processing.
///
/// ## How it Works:
/// 1.  **Initiate Request**: Sends the POST request to the server.
/// 2.  **Get ReadableStream**: Instead of awaiting the full response body with `.text().await`,
///     it immediately gets the `ReadableStream` from the response.
/// 3.  **Read Chunks**: It enters a loop, using a `ReadableStreamDefaultReader` to read
///     data chunks (`Uint8Array`) as soon as they are sent by the server.
/// 4.  **Decode and Buffer**: Each chunk is decoded from UTF-8 to a string and appended to a buffer.
///     This is crucial because a single chunk may not contain a complete SSE message.
/// 5.  **Process Lines**: The buffer is scanned for newline characters (`\n`), which delimit
///     SSE messages. Each complete line is processed immediately.
/// 6.  **Immediate UI Update**: When a valid "data: ..." message is parsed, it's immediately
///     dispatched to the Yew `Model` via `link.send_message()`, triggering an instant UI update
///     for that specific result.
/// 7.  **Completion**: The loop continues until the stream reports it's `done`, at which point
///     the loading state is reset.
async fn handle_streaming_request(
    endpoint: &str,
    form_data: FormData,
    auth_token: Option<String>,
    file_id_map: std::collections::HashMap<u64, u64>,
    link: yew::html::Scope<Model>,
    send_error: impl Fn(String) + 'static,
) {
    let mut request_builder = Request::post(endpoint);
    if let Some(token) = auth_token {
        request_builder = request_builder.header("Authorization", &format!("Bearer {}", token));
    }

    let request = match request_builder.body(form_data) {
        Ok(req) => req,
        Err(e) => {
            send_error(format!("Failed to build streaming request: {}", e));
            return;
        }
    };

    let response = match request.send().await {
        Ok(resp) => resp,
        Err(e) => {
            send_error(format!("Network error in streaming request: {}", e));
            return;
        }
    };

    if !response.ok() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown server error".to_string());
        send_error(format!("Server error in streaming: {}", error_text));
        return;
    }

    // Get the response body as a readable stream
    let body = match response.body() {
        Some(body) => body,
        None => {
            send_error("Response body is missing.".to_string());
            return;
        }
    };

    let reader: ReadableStreamDefaultReader = body.get_reader().dyn_into().unwrap();
    let decoder = TextDecoder::new().unwrap();
    let mut buffer = String::new();

    loop {
        match JsFuture::from(reader.read()).await {
            Ok(result) => {
                let result_obj: js_sys::Object = result.dyn_into().unwrap();
                let done_val = js_sys::Reflect::get(&result_obj, &"done".into()).unwrap();

                if done_val.as_bool().unwrap_or(true) {
                    log::info!("Streaming finished.");
                    break; // Exit loop when stream is done
                }

                let chunk_val = js_sys::Reflect::get(&result_obj, &"value".into()).unwrap();
                let chunk_uint8: js_sys::Uint8Array = chunk_val.dyn_into().unwrap();
                buffer.push_str(&decoder.decode_with_buffer_source(&chunk_uint8).unwrap());

                // Process all complete lines in the buffer
                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer.drain(..=newline_pos).collect::<String>();
                    let line = line.trim();

                    if line.starts_with("data: ") {
                        let data = &line[6..];

                        if data == "{\"type\": \"complete\"}" {
                            continue; // Skip the completion signal event
                        }

                        match serde_json::from_str::<Value>(data) {
                            Ok(event_data) => {
                                if let (Some(file_id_backend), Some(inference_data)) = (
                                    event_data.get("file_id").and_then(|v| v.as_u64()),
                                    event_data.get("inference"),
                                ) {
                                    if let Some(&frontend_file_id) =
                                        file_id_map.get(&file_id_backend)
                                    {
                                        match serde_json::from_value::<InferenceResponse>(
                                            inference_data.clone(),
                                        ) {
                                            Ok(response) => {
                                                log::info!(
                                                    "Received and processing result for file ID: {}",
                                                    frontend_file_id
                                                );
                                                link.send_message(Msg::InferenceResult(
                                                    frontend_file_id,
                                                    response,
                                                ));
                                            }
                                            Err(e) => log::error!(
                                                "Failed to parse streaming inference response: {}",
                                                e
                                            ),
                                        }
                                    }
                                } else if let Some(error) = event_data.get("error") {
                                    send_error(format!("Streaming error: {}", error));
                                    return; // Stop processing on error
                                }
                            }
                            Err(e) => log::error!("Failed to parse SSE event data: {}", e),
                        }
                    }
                }
            }
            Err(e) => {
                send_error(format!("Error reading from stream: {:?}", e));
                break;
            }
        }
    }

    // All results have been processed, reset the main loading state
    link.send_message(Msg::ResetLoadingState);
}

// No changes in this function
pub fn handle_inference_result(
    model: &mut Model,
    file_id: u64,
    response: InferenceResponse,
) -> bool {
    model.results.insert(file_id, response.clone());

    // Update the FileData to mark it as cached and set the image hash
    if let Some(file_data) = model.files.get_mut(&file_id) {
        if !file_data.is_cached && response.image_hash.is_some() {
            // This image was just processed and cached
            file_data.is_cached = true;
            file_data.image_hash = response.image_hash.clone();
            log::info!(
                "Marked file {} as cached with hash: {:?}",
                file_id,
                file_data.image_hash
            );
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
            } else {
                log::info!("Successfully cleared user cache");
            }
        });
    }

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
            model.results.remove(&file_id);

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

    model.loading = true;
    model.error = None;
    model.future_requests = model.files.len();
    model.batch_processing = true;  // This was missing!
    model.completed_requests = 0;   // Reset completed requests

    let files: Vec<_> = model.files.values().cloned().collect();
    inference_request_wrapper(ctx, files, model.processing_mode.clone());

    true
}

pub fn handle_set_processing_mode(
    model: &mut Model,
    _ctx: &Context<Model>,
    mode: ProcessingMode,
) -> bool {
    model.processing_mode = mode.clone();
    true
}

pub fn fetch_and_restore_session(ctx: &Context<Model>) {
    let link = ctx.link().clone();

    wasm_bindgen_futures::spawn_local(async move {
        let auth_token: Option<String> = LocalStorage::get("auth_token").ok();
        if auth_token.is_none() {
            log::info!("🔍 No auth token found, skipping session restoration");
            return;
        }

        log::info!("Fetching cached session data...");
        let mut request_builder = Request::get("/api/cache/history");

        // Add authorization header
        if let Some(token) = auth_token {
            request_builder = request_builder.header("Authorization", &format!("Bearer {}", token));
        }

        let response = match request_builder.send().await {
            Ok(resp) if resp.ok() => resp,
            Ok(resp) => {
                let status = resp.status();
                log::warn!("Failed to fetch session data: {}", status);
                return;
            }
            Err(e) => {
                log::warn!("Network error fetching session data: {}", e);
                return;
            }
        };

        let session_data: Value = match response.json().await {
            Ok(data) => data,
            Err(e) => {
                log::error!("Failed to parse session data: {}", e);
                return;
            }
        };

        log::info!("Session data received, restoring...");

        if let Some(images_obj) = session_data.get("images").and_then(|v| v.as_object()) {
            log::info!("Processing {} cached images", images_obj.len());
            for (file_id_str, image_data) in images_obj {
                if let Ok(file_id) = file_id_str.parse::<u64>() {
                    if let Some(preview_url) =
                        image_data.get("preview_url").and_then(|v| v.as_str())
                    {
                        // Create a mock file for the cached image
                        let file_name = image_data
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("cached_image");
                        let file_size =
                            image_data.get("size").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

                        // Create FileData and then set the preview URL
                        let image_hash = image_data
                            .get("image_hash")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let file_data = FileData {
                            id: file_id,
                            file: create_mock_file(file_name, file_size),
                            preview_url: None,
                            image_hash,
                            is_cached: true,
                        };

                        log::info!("Restoring cached file: {} ({})", file_id, file_name);
                        link.send_message(Msg::RestoreCachedFile(file_id, file_data.clone()));

                        // Fetch and set the preview URL
                        let link_clone = link.clone();
                        let preview_url_owned = preview_url.to_string();
                        spawn_local(async move {
                            // Create a blob URL from the cached image endpoint
                            match fetch_cached_image_as_blob(&preview_url_owned).await {
                                Ok(blob_url) => {
                                    log::info!("Created blob URL for cached image: {}", file_id);
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
            }
        }

        if let Some(results_obj) = session_data.get("results").and_then(|v| v.as_object()) {
            for (file_id_str, result_data) in results_obj {
                if let Ok(file_id) = file_id_str.parse::<u64>() {
                    if let Ok(inference_response) =
                        serde_json::from_value::<InferenceResponse>(result_data.clone())
                    {
                        link.send_message(Msg::InferenceResult(file_id, inference_response));
                    }
                }
            }
        }

        log::info!("Session restoration completed");
    });
}
// Helper function to create a mock file for cached images
fn create_mock_file(name: &str, size: u32) -> GlooFile {
    // Create a minimal blob to represent the cached file
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

// Helper function to fetch cached image and create a blob URL
async fn fetch_cached_image_as_blob(
    preview_url: &str,
) -> Result<ObjectUrl, Box<dyn std::error::Error>> {
    let auth_token: Option<String> = LocalStorage::get("auth_token").ok();
    if auth_token.is_none() {
        return Err("No auth token available".into());
    }

    let mut request_builder = Request::get(preview_url);
    if let Some(token) = auth_token {
        request_builder = request_builder.header("Authorization", &format!("Bearer {}", token));
    }

    let response = request_builder.send().await?;
    if !response.ok() {
        return Err(format!("Failed to fetch image: {}", response.status()).into());
    }

    let bytes = response.binary().await?;
    let uint8_array = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
    uint8_array.copy_from(&bytes);

    let blob = web_sys::Blob::new_with_u8_array_sequence(&js_sys::Array::of1(&uint8_array))
        .map_err(|e| format!("Failed to create blob: {:?}", e))?;

    let object_url = ObjectUrl::from(blob);
    Ok(object_url)
}

// Helper function to delete a cached image
async fn delete_cached_image(image_hash: &str) -> Result<(), Box<dyn std::error::Error>> {
    let auth_token: Option<String> = LocalStorage::get("auth_token").ok();
    if auth_token.is_none() {
        return Err("No auth token available".into());
    }

    let mut request_builder = Request::delete(&format!("/api/cache/image/{}", image_hash));
    if let Some(token) = auth_token {
        request_builder = request_builder.header("Authorization", &format!("Bearer {}", token));
    }

    let response = request_builder.send().await?;
    if !response.ok() {
        return Err(format!("Failed to delete cached image: {}", response.status()).into());
    }

    Ok(())
}

// Helper function to clear all cached data
async fn clear_user_cache() -> Result<(), Box<dyn std::error::Error>> {
    let auth_token: Option<String> = LocalStorage::get("auth_token").ok();
    if auth_token.is_none() {
        return Err("No auth token available".into());
    }

    let mut request_builder = Request::delete("/api/cache/clear");
    if let Some(token) = auth_token {
        request_builder = request_builder.header("Authorization", &format!("Bearer {}", token));
    }

    let response = request_builder.send().await?;
    if !response.ok() {
        return Err(format!("Failed to clear user cache: {}", response.status()).into());
    }

    Ok(())
}
