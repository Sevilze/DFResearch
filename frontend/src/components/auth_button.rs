use gloo_net::http::Request;
use gloo_storage::{LocalStorage, Storage};
use serde::{Deserialize, Serialize};
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub id: String,
    pub email: String,
    pub name: String,
    pub picture_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AuthState {
    Idle,
    Redirecting,
    Authenticating,
    FetchingUserInfo,
}

#[derive(Properties, PartialEq)]
pub struct AuthButtonProps {
    pub auth_token: Option<String>,
    pub on_token_change: Callback<Option<String>>,
}

#[function_component(AuthButton)]
pub fn auth_button(props: &AuthButtonProps) -> Html {
    let user_info = use_state(|| None::<UserInfo>);
    let auth_state = use_state(|| AuthState::Idle);
    let login_start_time = use_state(|| None::<f64>);
    let was_automatic_reauth = use_state(|| false);

    // Fetch user info when token is available
    {
        let user_info = user_info.clone();
        let auth_state = auth_state.clone();
        let login_start_time = login_start_time.clone();
        let was_automatic_reauth = was_automatic_reauth.clone();
        let token = props.auth_token.clone();
        let on_token_change = props.on_token_change.clone();

        use_effect_with(token.clone(), move |token_opt| {
            if let Some(token) = token_opt.clone() {
                log::info!("AuthButton: Processing token (length: {})", token.len());

                // Check if this is likely an automatic re-authentication
                let current_time = js_sys::Date::now();
                let is_automatic = if let Some(start_time) = *login_start_time {
                    let elapsed = current_time - start_time;
                    elapsed < 3000.0 // Less than 3 seconds indicates automatic re-auth
                } else {
                    false // No start time means this is from localStorage, not a fresh login
                };

                if is_automatic {
                    log::info!(
                        "AuthButton: Detected automatic re-authentication (elapsed: {}ms)",
                        current_time - login_start_time.as_ref().unwrap_or(&current_time)
                    );
                    was_automatic_reauth.set(true);
                    auth_state.set(AuthState::Authenticating);
                } else {
                    was_automatic_reauth.set(false);
                    auth_state.set(AuthState::FetchingUserInfo);
                }

                let user_info = user_info.clone();
                let auth_state = auth_state.clone();
                let on_token_change = on_token_change.clone();

                spawn_local(async move {
                    match fetch_user_info(&token).await {
                        Ok(info) => {
                            log::info!(
                                "AuthButton: User info fetched successfully: {}",
                                info.email
                            );
                            user_info.set(Some(info));
                            auth_state.set(AuthState::Idle);
                        }
                        Err(e) => {
                            log::error!("AuthButton: Failed to fetch user info: {:?}", e);
                            LocalStorage::delete("auth_token");
                            user_info.set(None);
                            auth_state.set(AuthState::Idle);
                            on_token_change.emit(None);
                        }
                    }
                });
            } else {
                log::info!("AuthButton: No token provided");
                user_info.set(None);
                auth_state.set(AuthState::Idle);
            }
        });
    }

    let handle_login = {
        let auth_state = auth_state.clone();
        let login_start_time = login_start_time.clone();
        Callback::from(move |_| {
            // Record when login process starts
            let start_time = js_sys::Date::now();
            login_start_time.set(Some(start_time));
            auth_state.set(AuthState::Redirecting);

            log::info!("AuthButton: Starting login process at {}", start_time);

            let window = web_sys::window().unwrap();
            let base_url = window.location().origin().unwrap().to_string();
            let login_url = format!("{}/auth/login", base_url);
            let _ = window.location().set_href(&login_url);
        })
    };

    let handle_logout = {
        let user_info = user_info.clone();
        let auth_state = auth_state.clone();
        let login_start_time = login_start_time.clone();
        let was_automatic_reauth = was_automatic_reauth.clone();
        let on_token_change = props.on_token_change.clone();
        Callback::from(move |_| {
            LocalStorage::delete("auth_token");
            user_info.set(None);
            auth_state.set(AuthState::Idle);
            login_start_time.set(None);
            was_automatic_reauth.set(false);
            on_token_change.emit(None);

            // Reload the page to clear any cached state
            let window = web_sys::window().unwrap();
            let _ = window.location().reload();
        })
    };

    // Render loading states based on authentication phase
    match &*auth_state {
        AuthState::Redirecting => {
            return html! {
                <div class="auth-button-container">
                    <div class="auth-loading redirecting">
                        <i class="fa-solid fa-arrow-right-from-bracket"></i>
                        {" Redirecting to Google..."}
                    </div>
                </div>
            };
        }
        AuthState::Authenticating => {
            let message = if *was_automatic_reauth {
                " Signing you in automatically..."
            } else {
                " Authenticating..."
            };
            return html! {
                <div class="auth-button-container">
                    <div class="auth-loading authenticating">
                        <i class="fa-solid fa-shield-halved fa-pulse"></i>
                        {message}
                    </div>
                </div>
            };
        }
        AuthState::FetchingUserInfo => {
            return html! {
                <div class="auth-button-container">
                    <div class="auth-loading fetching">
                        <i class="fa-solid fa-user fa-spin"></i>
                        {" Fetching your profile..."}
                    </div>
                </div>
            };
        }
        AuthState::Idle => {
            // Continue to normal rendering below
        }
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
                            {"Logout"}
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
                        {"Login"}
                    </button>
                </div>
            }
        }
    }
}

async fn fetch_user_info(token: &str) -> Result<UserInfo, Box<dyn std::error::Error>> {
    log::info!("Attempting to fetch user info with token");

    let response = Request::get("/auth/me")
        .header("Authorization", &format!("Bearer {}", token))
        .send()
        .await;

    match response {
        Ok(resp) if resp.ok() => {
            log::info!("/auth/me request successful");
            match resp.json::<UserInfo>().await {
                Ok(user_info) => {
                    log::info!("User info parsed successfully: {}", user_info.email);
                    Ok(user_info)
                }
                Err(e) => {
                    log::error!("Failed to parse user info JSON: {:?}", e);
                    Err(format!("Failed to parse user info: {:?}", e).into())
                }
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let error_text = resp.text().await.unwrap_or_default();
            log::error!("/auth/me failed with status {}: {}", status, error_text);
            Err(format!("Failed to fetch user info: {} - {}", status, error_text).into())
        }
        Err(e) => {
            log::error!("/auth/me network error: {:?}", e);
            Err(format!("Network error: {:?}", e).into())
        }
    }
}
