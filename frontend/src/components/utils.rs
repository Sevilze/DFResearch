use super::super::Model;
use gloo_file::File as GlooFile;
use gloo_timers::callback::Timeout;
use js_sys::Date;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use web_sys::FileList;
use yew::prelude::*;

pub fn generate_id() -> u64 {
    static ID_COUNTER: AtomicU64 = AtomicU64::new(0);
    let now = Date::now() as u64;
    let count = ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    now * 1000 + (count % 1000)
}

// Debounce function to limit button events
pub fn debounce<F>(duration: i32, callback: F) -> Callback<MouseEvent>
where
    F: Fn() + Clone + 'static,
{
    let timeout = Rc::new(RefCell::new(None::<Timeout>));
    let timeout_clone = Rc::clone(&timeout);

    Callback::from(move |_| {
        let mut timeout_ref = timeout_clone.borrow_mut();

        if let Some(old_timeout) = timeout_ref.take() {
            old_timeout.cancel();
        }

        let inner_callback = callback.clone();
        let new_timeout = Timeout::new(duration as u32, move || {
            inner_callback();
        });

        *timeout_ref = Some(new_timeout);
    })
}

pub fn extract_image_files(file_list: &FileList) -> Vec<GlooFile> {
    let files: Vec<GlooFile> = (0..file_list.length())
        .filter_map(|i| file_list.item(i))
        .filter(|file| file.type_().starts_with("image/"))
        .map(GlooFile::from)
        .collect();
    files
}

pub fn render_error_message(model: &Model) -> Html {
    if let Some(error_msg) = &model.error {
        html! {
            <div class="error-message">
                <i class="fa-solid fa-circle-exclamation"></i>
                <p>{ error_msg }</p>
            </div>
        }
    } else {
        html! {}
    }
}
