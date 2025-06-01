use gloo_storage::{LocalStorage, Storage};
use gloo_net::http::Request;
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub id: String,
    pub email: String,
    pub name: String,
    pub picture_url: Option<String>,
}

#[derive(Properties, PartialEq)]
pub struct AuthButtonProps {
    pub auth_token: Option<String>,
    pub on_token_change: Callback<Option<String>>,
}

#[function_component(AuthButton)]
pub fn auth_button(props: &AuthButtonProps) -> Html {
    let user_info = use_state(|| None::<UserInfo>);
    let loading = use_state(|| false);

    // Fetch user info when token is available
    {
        let user_info = user_info.clone();
        let loading = loading.clone();
        let token = props.auth_token.clone();
        let on_token_change = props.on_token_change.clone();

        use_effect_with(token.clone(), move |token_opt| {
            if let Some(token) = token_opt.clone() {
                log::info!("🔄 AuthButton: Processing token (length: {})", token.len());
                let user_info = user_info.clone();
                let loading = loading.clone();
                let on_token_change = on_token_change.clone();

                spawn_local(async move {
                    loading.set(true);

                    match fetch_user_info(&token).await {
                        Ok(info) => {
                            log::info!("✅ AuthButton: User info fetched successfully: {}", info.email);
                            user_info.set(Some(info));
                        }
                        Err(e) => {
                            log::error!("❌ AuthButton: Failed to fetch user info: {:?}", e);
                            // Token might be invalid, clear it
                            let _ = LocalStorage::delete("auth_token");
                            user_info.set(None);
                            on_token_change.emit(None);
                        }
                    }

                    loading.set(false);
                });
            } else {
                log::info!("🚫 AuthButton: No token provided");
                user_info.set(None);
            }
        });
    }

    let handle_login = Callback::from(|_| {
        let window = web_sys::window().unwrap();
        let _ = window.location().set_href("http://localhost:8081/auth/login");
    });

    let handle_logout = {
        let user_info = user_info.clone();
        let on_token_change = props.on_token_change.clone();
        Callback::from(move |_| {
            let _ = LocalStorage::delete("auth_token");
            user_info.set(None);
            on_token_change.emit(None);

            // Reload the page to clear any cached state
            let window = web_sys::window().unwrap();
            let _ = window.location().reload();
        })
    };

    if *loading {
        return html! {
            <div class="auth-button-container">
                <div class="auth-loading">
                    <i class="fa-solid fa-spinner fa-spin"></i>
                    {" Loading..."}
                </div>
            </div>
        };
    }

    match &*user_info {
        Some(user) => {
            html! {
                <div class="auth-button-container">
                    <div class="user-info">
                        <div class="user-details">
                            <span class="user-name">{&user.name}</span>
                            <span class="user-email">{&user.email}</span>
                        </div>
                        <button 
                            class="logout-button"
                            onclick={handle_logout}
                            title="Logout"
                        >
                            <i class="fa-solid fa-sign-out-alt"></i>
                            {" Logout"}
                        </button>
                    </div>
                </div>
            }
        }
        None => {
            html! {
                <div class="auth-button-container">
                    <button 
                        class="login-button"
                        onclick={handle_login}
                        title="Login with Google via AWS Cognito"
                    >
                        <i class="fa-brands fa-google"></i>
                        {" Login"}
                    </button>
                </div>
            }
        }
    }
}

async fn fetch_user_info(token: &str) -> Result<UserInfo, Box<dyn std::error::Error>> {
    log::info!("🔍 Attempting to fetch user info with token");

    let response = Request::get("/auth/me")
        .header("Authorization", &format!("Bearer {}", token))
        .send()
        .await;

    match response {
        Ok(resp) if resp.ok() => {
            log::info!("✅ /auth/me request successful");
            match resp.json::<UserInfo>().await {
                Ok(user_info) => {
                    log::info!("✅ User info parsed successfully: {}", user_info.email);
                    Ok(user_info)
                }
                Err(e) => {
                    log::error!("❌ Failed to parse user info JSON: {:?}", e);
                    Err(format!("Failed to parse user info: {:?}", e).into())
                }
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let error_text = resp.text().await.unwrap_or_default();
            log::error!("❌ /auth/me failed with status {}: {}", status, error_text);
            Err(format!("Failed to fetch user info: {} - {}", status, error_text).into())
        }
        Err(e) => {
            log::error!("❌ /auth/me network error: {:?}", e);
            Err(format!("Network error: {:?}", e).into())
        }
    }
}
