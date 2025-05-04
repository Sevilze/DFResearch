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
        let resp = Request::get(&format!("/api/tasks/{}", task_id))
            .send()
            .await
            .map_err(|err| {
                error!(format!("Fetch error: {:?}", err));
            })?;

        match resp.status() {
            202 => Ok(true),
            200 => {
                let fetched = resp.json::<Task>().await.map_err(|_| ())?;
                task.set(Some(fetched.clone()));
                
                let should_continue = fetched.status
                    .as_deref()
                    .map(|status| status == "pending")
                    .unwrap_or(false);
                
                Ok(should_continue)
            }
            _ => Ok(false)
        }
    }

    fn start_polling(
        task_id: String,
        task: UseStateHandle<Option<Task>>,
        polling: UseStateHandle<Option<Interval>>
    ) {
        let polling_clone = polling.clone();
        let interval = Interval::new(2000, move || {
            let task = task.clone();
            let polling = polling_clone.clone();
            let task_id = task_id.clone();
            
            spawn_local(async move {
                let should_continue = fetch_task_status(&task_id, task).await
                    .unwrap_or(false);
                
                if !should_continue {
                    polling.set(None);
                }
            });
        });
        
        polling.set(Some(interval));
    }

    {
        let task = task_clone.clone();
        let polling_outer = polling_clone.clone();

        use_effect_with_deps(
            move |task_id_opt| {
                // Reset state
                task.set(None);
                polling_outer.set(None);

                if let Some(task_id) = task_id_opt.clone() {
                    let task = task.clone();
                    let polling = polling_outer.clone();
                    
                    spawn_local(async move {
                        if let Ok(true) = fetch_task_status(&task_id, task.clone()).await {
                            start_polling(task_id, task, polling);
                        }
                    });
                }

                move || polling_outer.set(None)
            },
            task_id_for_closure,
        );
    }

    let render_task_content = |task: &Task| {
        html! {
            <>
                <p><strong>{"Task ID:"}</strong> {&task.id}</p>
                <p><strong>{"Status:"}</strong> {task.status.clone().unwrap_or_default()}</p>
                {
                    match (&task.result, &task.error) {
                        (Some(result), _) => html! { <pre>{to_string_pretty(result).unwrap_or_default()}</pre> },
                        (_, Some(error)) => html! { <p style="color:red;">{error}</p> },
                        _ => html! {}
                    }
                }
            </>
        }
    };

    html! {
        <div class="task-status">
            {
                match (&props.task_id, &*task_clone) {
                    (None, _) => html! {},
                    (Some(_), Some(task)) => render_task_content(task),
                    _ => html! { <p>{"Loading task status..."}</p> }
                }
            }
        </div>
    }
}
