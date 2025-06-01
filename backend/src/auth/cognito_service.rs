use aws_sdk_cognitoidentityprovider::Client as CognitoClient;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;
use url::Url;

#[derive(Error, Debug)]
pub enum CognitoError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("URL parsing failed: {0}")]
    UrlError(#[from] url::ParseError),
    #[error("AWS Cognito error: {0}")]
    AwsError(String),
    #[error("Invalid token: {0}")]
    InvalidToken(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CognitoTokenResponse {
    pub access_token: String,
    pub id_token: String,
    pub refresh_token: Option<String>,
    pub token_type: String,
    pub expires_in: u64,
}

// Custom deserializer for boolean values that might come as strings
fn deserialize_string_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Bool(b) => Ok(b),
        Value::String(s) => match s.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(serde::de::Error::custom(format!(
                "Invalid boolean string: {}",
                s
            ))),
        },
        _ => Err(serde::de::Error::custom("Expected boolean or string")),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CognitoUserInfo {
    pub sub: String,
    pub email: String,
    #[serde(deserialize_with = "deserialize_string_bool", default)]
    pub email_verified: bool,
    pub given_name: Option<String>,
    pub family_name: Option<String>,
    pub picture: Option<String>,
}

#[derive(Clone)]
pub struct CognitoService {
    http_client: HttpClient,
    client_id: String,
    client_secret: String,
    domain: String,
    redirect_uri: String,
    region: String,
}

impl CognitoService {
    pub fn new(
        _client: CognitoClient,
        _user_pool_id: String,
        client_id: String,
        client_secret: String,
        domain: String,
        redirect_uri: String,
        region: String,
    ) -> Self {
        Self {
            http_client: HttpClient::new(),
            client_id,
            client_secret,
            domain,
            redirect_uri,
            region,
        }
    }

    pub fn get_authorization_url(&self, state: &str) -> Result<String, CognitoError> {
        let mut url = Url::parse(&format!(
            "https://{}.auth.{}.amazoncognito.com/oauth2/authorize",
            self.domain, self.region
        ))?;

        url.query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", &self.client_id)
            .append_pair("redirect_uri", &self.redirect_uri)
            .append_pair("scope", "email openid profile")
            .append_pair("state", state)
            .append_pair("identity_provider", "Google");

        Ok(url.to_string())
    }

    pub async fn exchange_code_for_tokens(
        &self,
        code: &str,
    ) -> Result<CognitoTokenResponse, CognitoError> {
        let token_url = format!(
            "https://{}.auth.{}.amazoncognito.com/oauth2/token",
            self.domain, self.region
        );

        let mut params = HashMap::new();
        params.insert("grant_type", "authorization_code");
        params.insert("client_id", &self.client_id);
        params.insert("client_secret", &self.client_secret);
        params.insert("code", code);
        params.insert("redirect_uri", &self.redirect_uri);

        let response = self
            .http_client
            .post(&token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(CognitoError::AwsError(format!(
                "Token exchange failed: {}",
                error_text
            )));
        }

        let token_response: CognitoTokenResponse = response.json().await?;
        Ok(token_response)
    }

    pub async fn get_user_info(&self, access_token: &str) -> Result<CognitoUserInfo, CognitoError> {
        let user_info_url = format!(
            "https://{}.auth.{}.amazoncognito.com/oauth2/userInfo",
            self.domain, self.region
        );

        let response = self
            .http_client
            .get(&user_info_url)
            .bearer_auth(access_token)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(CognitoError::AwsError(format!(
                "User info request failed: {}",
                error_text
            )));
        }

        let user_info: CognitoUserInfo = response.json().await?;
        Ok(user_info)
    }

    pub async fn refresh_token(
        &self,
        refresh_token: &str,
    ) -> Result<CognitoTokenResponse, CognitoError> {
        // Validate refresh token format
        if refresh_token.is_empty() {
            return Err(CognitoError::InvalidToken("Empty refresh token".to_string()));
        }
        let token_url = format!(
            "https://{}.auth.{}.amazoncognito.com/oauth2/token",
            self.domain, self.region
        );

        let mut params = HashMap::new();
        params.insert("grant_type", "refresh_token");
        params.insert("client_id", &self.client_id);
        params.insert("client_secret", &self.client_secret);
        params.insert("refresh_token", refresh_token);

        let response = self
            .http_client
            .post(&token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(CognitoError::AwsError(format!(
                "Token refresh failed: {}",
                error_text
            )));
        }

        let token_response: CognitoTokenResponse = response.json().await?;
        Ok(token_response)
    }

    pub async fn revoke_token(&self, token: &str) -> Result<(), CognitoError> {
        let revoke_url = format!(
            "https://{}.auth.{}.amazoncognito.com/oauth2/revoke",
            self.domain, self.region
        );

        let mut params = HashMap::new();
        params.insert("token", token);
        params.insert("client_id", &self.client_id);
        params.insert("client_secret", &self.client_secret);

        let response = self
            .http_client
            .post(&revoke_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(CognitoError::AwsError(format!(
                "Token revocation failed: {}",
                error_text
            )));
        }

        Ok(())
    }
}
