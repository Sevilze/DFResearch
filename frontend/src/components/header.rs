use yew::prelude::*;

pub fn render_header() -> Html {
    html! {
        <header class="app-header">
            <h1><i class="fa-solid fa-images"></i> {" Multi-Image Upload & Analysis"}</h1>
            <p class="subtitle">{"Upload images via button, drag & drop, or paste"}</p>
        </header>
    }
}
