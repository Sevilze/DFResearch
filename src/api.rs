use yew::prelude::*;
use gloo_net::http::Request;
use gloo_timers::callback::Interval;
use shared::Task;

#[derive(Properties, PartialEq)]
pub struct TaskStatusProps {
    pub task_id: Option<String>,
}

#[function_component(TaskStatus)]
pub fn task_status(props: &TaskStatusProps) -> Html {
    let task = use_state(|| None::<Task>);
    let polling = use_state(|| None::<Interval>);

    let task_id_for_closure = props.task_id.clone();

    {
        let task_id_inner = task_id_for_closure.clone();
        let task = task.clone();
        let polling_outer = polling.clone();
        let polling = polling.clone();

        use_effect_with_deps(
            move |_| {
                // Cleanup any existing polling
                polling_outer.set(None);

                if let Some(task_id_inner) = task_id_inner {
                    // Fetch once immediately
                    wasm_bindgen_futures::spawn_local({
                        let task = task.clone();
                        let polling_clone = polling.clone();
                        let polling_clone_for_poll = polling_clone.clone();
                        let task_id_clone = task_id_inner.clone();
                        async move {
                            gloo_console::log!("Fetching task ID once:", &task_id_clone);
                            match Request::get(&format!("/api/tasks/{}", task_id_clone))
                                .send()
                                .await
                            {
                                Ok(resp) => {
                                    let status_code = resp.status();
                                    if status_code == 202 {
                                        gloo_console::log!("Task not yet created, starting polling");
                                        let interval = Interval::new(2000, move || {
                                            let task = task.clone();
                                            let task_id_clone2 = task_id_clone.clone();
                                            let polling_clone2 = polling_clone_for_poll.clone();

                                            wasm_bindgen_futures::spawn_local(async move {
                                                gloo_console::log!("Polling task ID:", &task_id_clone2);
                                                match Request::get(&format!("/api/tasks/{}", task_id_clone2))
                                                    .send()
                                                    .await
                                                {
                                                    Ok(resp) => {
                                                        let status_code = resp.status();
                                                        if status_code == 202 {
                                                        } else if status_code == 200 {
                                                            if let Ok(fetched) = resp.json::<Task>().await {
                                                                gloo_console::log!("Fetched task:", format!("{:?}", fetched));
                                                                task.set(Some(fetched.clone()));
                                                                if let Some(status) = &fetched.status {
                                                                    if status == "completed" || status == "failed" {
                                                                        polling_clone2.set(None);
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            polling_clone2.set(None);
                                                        }
                                                    }
                                                    Err(err) => {
                                                        gloo_console::error!(format!("Fetch error: {:?}", err));
                                                    }
                                                }
                                            });
                                        });
                                        polling_clone.set(Some(interval));
                                    } else if status_code == 200 {
                                        if let Ok(fetched) = resp.json::<Task>().await {
                                            gloo_console::log!("Fetched task:", format!("{:?}", fetched));
                                            task.set(Some(fetched.clone()));
                                            if let Some(status) = &fetched.status {
                                                if status == "pending" {
                                                    let interval = Interval::new(2000, move || {
                                                        let task = task.clone();
                                                        let task_id_clone2 = task_id_clone.clone();
                                                        let polling_clone2 = polling_clone_for_poll.clone();

                                                        wasm_bindgen_futures::spawn_local(async move {
                                                            gloo_console::log!("Polling task ID:", &task_id_clone2);
                                                            match Request::get(&format!("/api/tasks/{}", task_id_clone2))
                                                                .send()
                                                                .await
                                                            {
                                                                Ok(resp) => {
                                                                    let status_code = resp.status();
                                                                    if status_code == 202 {
                                                                    } else if status_code == 200 {
                                                                        if let Ok(fetched) = resp.json::<Task>().await {
                                                                            gloo_console::log!("Fetched task:", format!("{:?}", fetched));
                                                                            task.set(Some(fetched.clone()));
                                                                            if let Some(status) = &fetched.status {
                                                                                if status == "completed" || status == "failed" {
                                                                                    polling_clone2.set(None);
                                                                                }
                                                                            }
                                                                        }
                                                                    } else {
                                                                        polling_clone2.set(None);
                                                                    }
                                                                }
                                                                Err(err) => {
                                                                    gloo_console::error!(format!("Fetch error: {:?}", err));
                                                                }
                                                            }
                                                        });
                                                    });
                                                    polling_clone.set(Some(interval));
                                                }
                                            }
                                        }
                                    }
                                }
                                Err(err) => {
                                    gloo_console::error!(format!("Fetch error: {:?}", err));
                                }
                            }
                        }
                    });
                }

                move || {
                    polling_outer.set(None);
                }
            },
            task_id_for_closure.clone(),
        );
    }

    html! {
        <div class="task-status">
            {
                if props.task_id.is_none() {
                    html! {}
                } else if let Some(task) = &*task {
                    html! {
                        <>
                            <p><strong>{"Task ID:"}</strong> {&task.id}</p>
                            <p><strong>{"Status:"}</strong> {task.status.clone().unwrap_or_default()}</p>
                            {
                                if let Some(result) = &task.result {
                                    html! { <pre>{serde_json::to_string_pretty(result).unwrap_or_default()}</pre> }
                                } else if let Some(error) = &task.error {
                                    html! { <p style="color:red;">{error}</p> }
                                } else {
                                    html! {}
                                }
                            }
                        </>
                    }
                } else {
                    html! { <p>{"Loading task status..."}</p> }
                }
            }
        </div>
    }
}
