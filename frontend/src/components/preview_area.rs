use super::super::{FileData, Model, Msg};
use super::utils::debounce;
use shared::ProcessingMode; // Import ProcessingMode
use yew::prelude::*;

pub fn render_preview_area(model: &Model, ctx: &Context<Model>) -> Html {
    if model.files.is_empty() {
        return html! {};
    }

    let link = ctx.link().clone();

    html! {
        <div id="preview-container">
            { render_selected_image_preview(model) }
            <h2>{ format!("Previews: {} / 15", model.files.len()) }</h2>
            <div id="image-previews">
                {{
                    let mut sorted_files: Vec<&FileData> = model.files.values().collect();
                    sorted_files.sort_by_key(|fd| fd.id);
                    sorted_files.iter()
                        .map(|file_data| render_preview_item(ctx, model, file_data))
                        .collect::<Html>()
                }}
            </div>
            <div class="processing-mode-selector">
                <label>
                    <input type="radio" name="processing_mode"
                           value="intermediate"
                            checked={model.processing_mode == ProcessingMode::IntermediateFusionEnsemble}
                            onchange={ctx.link().callback(|_| Msg::SetProcessingMode(ProcessingMode::IntermediateFusionEnsemble))} />
                     <span class="radio-label-text">{ "Intermediate Fusion" }</span>
                 </label>
                 <label>
                    <input type="radio" name="processing_mode"
                           value="late"
                            checked={model.processing_mode == ProcessingMode::LateFusionEnsemble}
                            onchange={ctx.link().callback(|_| Msg::SetProcessingMode(ProcessingMode::LateFusionEnsemble))} />
                     <span class="radio-label-text">{ "Late Fusion" }</span>
                 </label>
             </div>
            <div class="button-container">
                <button
                    id="clear-all-btn"
                    class="analyze-btn"
                    style="background-color: var(--clear-color);"
                    onclick={debounce(300, {
                        let link = link.clone();
                        move || link.send_message(Msg::ClearAllFiles)
                    })}
                >
                    <i class="fa-solid fa-trash"></i>{" Clear All"}
                </button>
                <button
                    class="analyze-btn"
                    onclick={debounce(300, {
                        let link = link.clone();
                        move || link.callback(|_| Msg::AnalyzeSelected).emit(())
                    })}
                    disabled={model.loading || model.selected_file_id.is_none()}
                >
                    { render_analyze_button_content(model) }
                </button>
                <button
                    class="analyze-btn"
                    style="background-color: var(--primary-color);"
                    onclick={debounce(300, {
                        let link = link.clone();
                        move || link.callback(|_| Msg::AnalyzeAll).emit(())
                    })}
                >
                    <i class="fa-solid fa-magnifying-glass"></i>{" Analyze All"}
                </button>
            </div>
        </div>
    }
}

fn render_preview_item(ctx: &Context<Model>, model: &Model, file_data: &FileData) -> Html {
    let file_id = file_data.id;
    let link = ctx.link();
    let is_selected = model.selected_file_id == Some(file_id);

    html! {
        <div
            class={classes!("preview-item", is_selected.then_some("selected"))}
            key={file_id.to_string()}
            onclick={link.callback(move |_| Msg::SelectFile(file_id))}
            title={format!("Click to select for analysis: {}", file_data.file.name())}
        >
            {
                if let Some(url) = &file_data.preview_url {
                    html! { <img src={url.to_string()} alt={file_data.file.name()} /> }
                } else {
                    html! { <div class="preview-placeholder preview-placeholder-centered">{"..."}</div> }
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

fn render_selected_image_preview(model: &Model) -> Html {
    match model.selected_file_id {
        Some(_id) if model.preview_loading => html! {
            <div class="loading-preview">
                <i class="fa-solid fa-spinner fa-spin fa-2x"></i>
                <p style="margin-left: 10px;">{"Loading preview..."}</p>
            </div>
        },
        Some(id) => {
            if let Some(url) = model.files.get(&id).and_then(|fd| fd.preview_url.as_ref()) {
                html! {
                    <img id="actual-image-preview"
                        src={url.to_string()}
                        alt="Image Preview" />
                }
            } else {
                html! {
                    <div class="unavailable-preview">
                        <p>{"Preview unavailable"}</p>
                    </div>
                }
            }
        }
        None => html! {
            <div class="select-preview">
                <p>{"Select an image preview below"}</p>
            </div>
        },
    }
}

fn render_analyze_button_content(model: &Model) -> Html {
    if model.loading {
        html! { <><i class="fa-solid fa-spinner fa-spin"></i>{" Analyzing..."}</> }
    } else {
        let filename = model
            .selected_file_id
            .and_then(|id| model.files.get(&id))
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
