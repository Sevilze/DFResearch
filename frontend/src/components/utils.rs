use yew::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;
use js_sys::Date;
use gloo_timers::callback::Timeout;
use super::super::Model;

pub fn generate_id() -> u64 {
    (Date::new_0().get_time() * 1000.0 + js_sys::Math::random() * 1000.0) as u64
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

