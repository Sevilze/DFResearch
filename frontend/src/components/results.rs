use super::super::Model;
use yew::prelude::*;

pub fn render_results(model: &Model) -> Html {
    if let Some(selected_id) = model.selected_file_id {
        if let Some(results) = model.results.get(&selected_id) {
            let predicted_class = results
                .predictions
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let confidence = results.predictions[predicted_class] * 100.0;
            let is_ai = predicted_class == 0;
            let analyzed_filename = model
                .files
                .get(&selected_id)
                .map_or_else(|| "Analyzed Image".to_string(), |fd| fd.file.name());

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
            html! { <p class="no-results-message">{"No analysis result available for the selected image."}</p> }
        }
    } else {
        html! {}
    }
}
