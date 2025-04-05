use yew::prelude::*;
use super::super::Model;
use super::super::Msg;
use yew::html::Scope;

pub fn render_theme_toggle(theme: &str, link: &Scope<Model>) -> Html {
    html! {
        <div class="top-right">
            <button
                id="theme-toggle"
                class="theme-toggle"
                onclick={link.callback(|_| Msg::ToggleTheme)}
                title={ if theme == "light" { "Switch to Dark Mode" } else { "Switch to Light Mode" } }
            >
                { if theme == "light" {
                    html! { <img src="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/2600.svg" alt="Sun Icon" class="toggle-icon" /> }
                } else {
                    html! { <img src="https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/svg/1f319.svg" alt="Moon Icon" class="toggle-icon" /> }
                }}
            </button>
        </div>
    }
}
