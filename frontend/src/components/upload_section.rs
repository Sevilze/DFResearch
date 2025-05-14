use super::super::Model;
use super::super::Msg;
use super::utils::{debounce, extract_image_files};
use wasm_bindgen::JsCast;
use web_sys::{DragEvent, HtmlInputElement};
use yew::prelude::*;

pub fn render_upload_section(model: &Model, ctx: &Context<Model>) -> Html {
    let limit_reached = model.files.len() >= 15;
    html! {
        <div class="upload-section">
            { render_file_input_area(model, ctx, limit_reached) }
        </div>
    }
}

fn render_file_input_area(model: &Model, ctx: &Context<Model>, limit_reached: bool) -> Html {
    if limit_reached {
        return html! {
            <p class="limit-reached">{"You have reached the maximum of 15 images."}</p>
        };
    }

    let link = ctx.link();
    let handle_change = link.callback(|e: Event| {
        let input: HtmlInputElement = e.target_unchecked_into();
        let files = input.files();
        let files_to_process = files.as_ref().map(extract_image_files).unwrap_or_default();

        input.set_value("");

        if !files_to_process.is_empty() {
            Msg::FilesAdded(files_to_process)
        } else {
            Msg::SetError(Some("No valid image files selected.".into()))
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
        if let Some(input) = web_sys::window()
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
                onclick={debounce(300, {
                    let trigger_file_input = trigger_file_input.clone();
                    move || trigger_file_input.emit(())
                })}
            >
                <i class="fa-solid fa-upload"></i> {" Select Images"}
            </button>

            <div
                id="drop-zone"
                class={classes!("upload-area", model.is_dragging.then_some("drag-over"))}
                ondragover={handle_drag_over}
                ondragleave={handle_drag_leave}
                ondrop={handle_drop}
                onclick={debounce(300, {
                    let trigger_file_input = trigger_file_input.clone();
                    move || trigger_file_input.emit(())
                })}
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
