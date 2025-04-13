use yew::prelude::*;
use gloo_net::http::Request;
use gloo_console::error;
use gloo_timers::callback::Interval;
use wasm_bindgen_futures::spawn_local;
use serde_json::to_string_pretty;
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
    let task_clone = task.clone();
    let polling_clone = polling.clone();

    async fn fetch_task_status(task_id: &str, task: UseStateHandle<Option<Task>>) -> Result<bool, ()> {
        match Request::get(&format!("/api/tasks/{}", task_id)).send().await {
            Ok(resp) => {
                let status_code = resp.status();
                if status_code == 202 {
                    Ok(true)
                } else if status_code == 200 {
                    if let Ok(fetched) = resp.json::<Task>().await {
                        task.set(Some(fetched.clone()));
                        if let Some(status) = &fetched.status {
                            if status == "completed" || status == "failed" {
                                return Ok(false);
                            } else if status == "pending" {
                                return Ok(true);
                            }
                        }
                        Ok(false)
                    } else {
                        Ok(false)
                    }
                } else {
                    Ok(false)
                }
            }
            Err(err) => {
                error!(format!("Fetch error: {:?}", err));
                Err(())
            }
        }
    }

    fn start_polling(task_id: String, task: UseStateHandle<Option<Task>>, polling: UseStateHandle<Option<Interval>>) {
        let interval = Interval::new(2000, {
            let task = task.clone();
            let polling = polling.clone();
            let task_id = task_id.clone();
            move || {
                let task = task.clone();
                let polling = polling.clone();
                let task_id = task_id.clone();
                spawn_local(async move {
                    match fetch_task_status(&task_id, task.clone()).await {
                        Ok(continue_polling) => {
                            if !continue_polling {
                                polling.set(None);
                            }
                        }
                        Err(_) => {
                            polling.set(None);
                        }
                    }
                });
            }
        });
        polling.set(Some(interval));
    }

    {
        let task = task_clone.clone();
        let polling_outer = polling_clone.clone();

        use_effect_with_deps(
            move |task_id_opt| {
                task.set(None);
                polling_outer.set(None);

                if let Some(task_id) = task_id_opt.clone() {
                    let task = task.clone();
                    let polling = polling_outer.clone();
                    spawn_local(async move {
                        match fetch_task_status(&task_id, task.clone()).await {
                            Ok(continue_polling) => {
                                if continue_polling {
                                    start_polling(task_id, task, polling);
                                }
                            }
                            Err(_) => {
                            }
                        }
                    });
                }

                move || {
                    polling_outer.set(None);
                }
            },
            task_id_for_closure,
        );
    }

    html! {
        <div class="task-status">
            {
                if props.task_id.is_none() {
                    html! {}
                } else if let Some(task) = &*task_clone {
                    html! {
                        <>
                            <p><strong>{"Task ID:"}</strong> {&task.id}</p>
                            <p><strong>{"Status:"}</strong> {task.status.clone().unwrap_or_default()}</p>
                            {
                                if let Some(result) = &task.result {
                                    html! { <pre>{to_string_pretty(result).unwrap_or_default()}</pre> }
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
