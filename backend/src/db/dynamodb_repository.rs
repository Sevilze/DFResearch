use aws_sdk_dynamodb::types::AttributeValue;
use aws_sdk_dynamodb::Client;
use chrono::{DateTime, Utc};
use serde_json;
use std::collections::HashMap;
use uuid::Uuid;

use crate::cache::models::{ImageCacheEntry, InferenceCacheEntry, User};
use shared::ProcessingMode;

#[derive(Clone)]
pub struct DynamoDbRepository {
    client: Client,
    users_table: String,
    images_table: String,
    inference_table: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RepositoryError {
    #[error("DynamoDB error: {0}")]
    DynamoDb(String),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Item not found")]
    NotFound,
    #[error("Invalid data format: {0}")]
    InvalidData(String),
}

impl DynamoDbRepository {
    pub fn new(
        client: Client,
        users_table: String,
        images_table: String,
        inference_table: String,
    ) -> Self {
        Self {
            client,
            users_table,
            images_table,
            inference_table,
        }
    }

    pub async fn create_user(&self, user: &User) -> Result<(), RepositoryError> {
        log::info!(
            "🔄 Creating user in DynamoDB table '{}': {}",
            self.users_table,
            user.email
        );

        // Verify the table exists
        match self
            .client
            .describe_table()
            .table_name(&self.users_table)
            .send()
            .await
        {
            Ok(response) => {
                log::info!(
                    "DynamoDB table '{}' exists with status: {:?}",
                    self.users_table,
                    response.table().and_then(|t| t.table_status())
                );
            }
            Err(e) => {
                log::error!(
                    "DynamoDB table '{}' does not exist or is not accessible: {:?}",
                    self.users_table,
                    e
                );
                return Err(RepositoryError::DynamoDb(format!(
                    "Table '{}' not accessible: {}",
                    self.users_table, e
                )));
            }
        }

        let mut item = HashMap::new();
        item.insert("id".to_string(), AttributeValue::S(user.id.to_string()));

        // Cognito fields
        item.insert(
            "cognito_sub".to_string(),
            AttributeValue::S(user.cognito_sub.clone()),
        );
        if let Some(cognito_access_token) = &user.cognito_access_token {
            item.insert(
                "cognito_access_token".to_string(),
                AttributeValue::S(cognito_access_token.clone()),
            );
        }
        if let Some(cognito_refresh_token) = &user.cognito_refresh_token {
            item.insert(
                "cognito_refresh_token".to_string(),
                AttributeValue::S(cognito_refresh_token.clone()),
            );
        }

        // Common fields
        item.insert("email".to_string(), AttributeValue::S(user.email.clone()));
        item.insert("name".to_string(), AttributeValue::S(user.name.clone()));

        if let Some(picture_url) = &user.picture_url {
            item.insert(
                "picture_url".to_string(),
                AttributeValue::S(picture_url.clone()),
            );
        }

        item.insert(
            "created_at".to_string(),
            AttributeValue::S(user.created_at.to_rfc3339()),
        );
        item.insert(
            "updated_at".to_string(),
            AttributeValue::S(user.updated_at.to_rfc3339()),
        );

        if let Some(last_login) = &user.last_login {
            item.insert(
                "last_login".to_string(),
                AttributeValue::S(last_login.to_rfc3339()),
            );
        }

        item.insert(
            "is_active".to_string(),
            AttributeValue::Bool(user.is_active),
        );

        match self
            .client
            .put_item()
            .table_name(&self.users_table)
            .set_item(Some(item))
            .send()
            .await
        {
            Ok(_) => {
                log::info!("✅ Successfully created user in DynamoDB: {}", user.email);
                Ok(())
            }
            Err(e) => {
                log::error!("DynamoDB put_item failed for user {}: {:?}", user.email, e);
                Err(RepositoryError::DynamoDb(e.to_string()))
            }
        }
    }

    pub async fn get_user_by_id(&self, user_id: Uuid) -> Result<Option<User>, RepositoryError> {
        let mut key = HashMap::new();
        key.insert("id".to_string(), AttributeValue::S(user_id.to_string()));

        let result = self
            .client
            .get_item()
            .table_name(&self.users_table)
            .set_key(Some(key))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        if let Some(item) = result.item {
            Ok(Some(self.parse_user_from_item(item)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_user(&self, user_id: &Uuid) -> Result<User, RepositoryError> {
        match self.get_user_by_id(*user_id).await? {
            Some(user) => Ok(user),
            None => Err(RepositoryError::NotFound),
        }
    }

    pub async fn get_user_by_cognito_sub(
        &self,
        cognito_sub: &str,
    ) -> Result<User, RepositoryError> {
        let result = self
            .client
            .scan()
            .table_name(&self.users_table)
            .filter_expression("cognito_sub = :cognito_sub")
            .expression_attribute_values(":cognito_sub", AttributeValue::S(cognito_sub.to_string()))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        if let Some(items) = result.items {
            if let Some(item) = items.into_iter().next() {
                return self.parse_user_from_item(item);
            }
        }
        Err(RepositoryError::NotFound)
    }

    pub async fn update_user(&self, user: &User) -> Result<(), RepositoryError> {
        log::info!("Updating user in DynamoDB: {}", user.email);

        let mut key = HashMap::new();
        key.insert("id".to_string(), AttributeValue::S(user.id.to_string()));

        let mut update_expression_parts = Vec::new();
        let mut expression_attribute_values = HashMap::new();
        let mut expression_attribute_names = HashMap::new();

        // Update Cognito fields
        if let Some(cognito_access_token) = &user.cognito_access_token {
            update_expression_parts.push("cognito_access_token = :cognito_access_token");
            expression_attribute_values.insert(
                ":cognito_access_token".to_string(),
                AttributeValue::S(cognito_access_token.clone()),
            );
            log::debug!("Adding cognito_access_token to update");
        }

        if let Some(cognito_refresh_token) = &user.cognito_refresh_token {
            update_expression_parts.push("cognito_refresh_token = :cognito_refresh_token");
            expression_attribute_values.insert(
                ":cognito_refresh_token".to_string(),
                AttributeValue::S(cognito_refresh_token.clone()),
            );
            log::debug!("Adding cognito_refresh_token to update");
        }

        // Update common fields
        update_expression_parts.push("email = :email");
        expression_attribute_values
            .insert(":email".to_string(), AttributeValue::S(user.email.clone()));

        update_expression_parts.push("#name = :name");
        expression_attribute_names.insert("#name".to_string(), "name".to_string());
        expression_attribute_values
            .insert(":name".to_string(), AttributeValue::S(user.name.clone()));

        if let Some(picture_url) = &user.picture_url {
            update_expression_parts.push("picture_url = :picture_url");
            expression_attribute_values.insert(
                ":picture_url".to_string(),
                AttributeValue::S(picture_url.clone()),
            );
        }

        update_expression_parts.push("updated_at = :updated_at");
        expression_attribute_values.insert(
            ":updated_at".to_string(),
            AttributeValue::S(user.updated_at.to_rfc3339()),
        );

        if let Some(last_login) = &user.last_login {
            update_expression_parts.push("last_login = :last_login");
            expression_attribute_values.insert(
                ":last_login".to_string(),
                AttributeValue::S(last_login.to_rfc3339()),
            );
        }

        update_expression_parts.push("is_active = :is_active");
        expression_attribute_values.insert(
            ":is_active".to_string(),
            AttributeValue::Bool(user.is_active),
        );

        let update_expression = format!("SET {}", update_expression_parts.join(", "));

        log::debug!("Update expression: {}", update_expression);
        log::debug!(
            "Expression attribute names: {:?}",
            expression_attribute_names
        );
        log::debug!(
            "Expression attribute values: {:?}",
            expression_attribute_values
        );

        let mut update_request = self
            .client
            .update_item()
            .table_name(&self.users_table)
            .set_key(Some(key))
            .update_expression(update_expression)
            .set_expression_attribute_values(Some(expression_attribute_values));

        if !expression_attribute_names.is_empty() {
            update_request =
                update_request.set_expression_attribute_names(Some(expression_attribute_names));
        }

        match update_request.send().await {
            Ok(response) => {
                log::info!("Successfully updated user: {}", user.email);
                log::debug!("Update response: {:?}", response);
                Ok(())
            }
            Err(e) => {
                log::error!(
                    "DynamoDB update_item failed for user {}: {:?}",
                    user.email,
                    e
                );

                let error_msg = if let Some(service_err) = e.as_service_error() {
                    format!("Service error: {:?}", service_err)
                } else {
                    format!("SDK error: {}", e)
                };

                Err(RepositoryError::DynamoDb(error_msg))
            }
        }
    }

    // Image cache operations
    pub async fn create_image_cache(&self, entry: &ImageCacheEntry) -> Result<(), RepositoryError> {
        let mut item = HashMap::new();
        item.insert("id".to_string(), AttributeValue::S(entry.id.to_string()));
        item.insert(
            "user_id".to_string(),
            AttributeValue::S(entry.user_id.to_string()),
        );
        item.insert(
            "image_hash".to_string(),
            AttributeValue::S(entry.image_hash.clone()),
        );
        item.insert(
            "file_name".to_string(),
            AttributeValue::S(entry.file_name.clone()),
        );
        item.insert(
            "file_size".to_string(),
            AttributeValue::N(entry.file_size.to_string()),
        );
        item.insert(
            "mime_type".to_string(),
            AttributeValue::S(entry.mime_type.clone()),
        );
        item.insert(
            "s3_key".to_string(),
            AttributeValue::S(entry.s3_key.clone()),
        );
        item.insert(
            "s3_bucket".to_string(),
            AttributeValue::S(entry.s3_bucket.clone()),
        );
        item.insert(
            "created_at".to_string(),
            AttributeValue::S(entry.created_at.to_rfc3339()),
        );
        item.insert(
            "last_accessed".to_string(),
            AttributeValue::S(entry.last_accessed.to_rfc3339()),
        );
        item.insert(
            "access_count".to_string(),
            AttributeValue::N(entry.access_count.to_string()),
        );

        self.client
            .put_item()
            .table_name(&self.images_table)
            .set_item(Some(item))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        Ok(())
    }

    pub async fn get_image_cache_by_hash(
        &self,
        user_id: Uuid,
        image_hash: &str,
    ) -> Result<Option<ImageCacheEntry>, RepositoryError> {
        let result = self
            .client
            .scan()
            .table_name(&self.images_table)
            .filter_expression("user_id = :user_id AND image_hash = :image_hash")
            .expression_attribute_values(":user_id", AttributeValue::S(user_id.to_string()))
            .expression_attribute_values(":image_hash", AttributeValue::S(image_hash.to_string()))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        if let Some(items) = result.items {
            if let Some(item) = items.into_iter().next() {
                return Ok(Some(self.parse_image_cache_from_item(item)?));
            }
        }
        Ok(None)
    }

    pub async fn get_user_image_cache(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<ImageCacheEntry>, RepositoryError> {
        let result = self
            .client
            .scan()
            .table_name(&self.images_table)
            .filter_expression("user_id = :user_id")
            .expression_attribute_values(":user_id", AttributeValue::S(user_id.to_string()))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        let mut entries = Vec::new();
        if let Some(items) = result.items {
            for item in items {
                entries.push(self.parse_image_cache_from_item(item)?);
            }
        }
        Ok(entries)
    }

    pub async fn delete_image_cache(&self, entry_id: Uuid) -> Result<(), RepositoryError> {
        let mut key = HashMap::new();
        key.insert("id".to_string(), AttributeValue::S(entry_id.to_string()));

        self.client
            .delete_item()
            .table_name(&self.images_table)
            .set_key(Some(key))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        Ok(())
    }

    pub async fn delete_image_cache_by_hash(
        &self,
        user_id: Uuid,
        image_hash: &str,
    ) -> Result<(), RepositoryError> {
        if let Some(entry) = self.get_image_cache_by_hash(user_id, image_hash).await? {
            self.delete_image_cache(entry.id).await?;
        }
        Ok(())
    }

    pub async fn delete_user_image_cache(&self, user_id: Uuid) -> Result<(), RepositoryError> {
        let entries = self.get_user_image_cache(user_id).await?;

        for entry in entries {
            self.delete_image_cache(entry.id).await?;
        }

        Ok(())
    }

    // Inference cache operations
    pub async fn create_inference_cache(
        &self,
        entry: &InferenceCacheEntry,
    ) -> Result<(), RepositoryError> {
        let mut item = HashMap::new();
        item.insert("id".to_string(), AttributeValue::S(entry.id.to_string()));
        item.insert(
            "user_id".to_string(),
            AttributeValue::S(entry.user_id.to_string()),
        );
        item.insert(
            "image_hash".to_string(),
            AttributeValue::S(entry.image_hash.clone()),
        );
        item.insert(
            "processing_mode".to_string(),
            AttributeValue::S(entry.processing_mode.clone()),
        );
        item.insert(
            "predictions".to_string(),
            AttributeValue::S(entry.predictions.to_string()),
        );
        item.insert("is_ai".to_string(), AttributeValue::Bool(entry.is_ai));
        item.insert(
            "confidence".to_string(),
            AttributeValue::N(entry.confidence.to_string()),
        );
        item.insert(
            "created_at".to_string(),
            AttributeValue::S(entry.created_at.to_rfc3339()),
        );
        item.insert(
            "last_accessed".to_string(),
            AttributeValue::S(entry.last_accessed.to_rfc3339()),
        );
        item.insert(
            "access_count".to_string(),
            AttributeValue::N(entry.access_count.to_string()),
        );

        self.client
            .put_item()
            .table_name(&self.inference_table)
            .set_item(Some(item))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        Ok(())
    }

    pub async fn get_inference_cache(
        &self,
        user_id: Uuid,
        image_hash: &str,
        processing_mode: &ProcessingMode,
    ) -> Result<Option<InferenceCacheEntry>, RepositoryError> {
        let mode_str = format!("{:?}", processing_mode);

        let result = self.client
            .scan()
            .table_name(&self.inference_table)
            .filter_expression("user_id = :user_id AND image_hash = :image_hash AND processing_mode = :processing_mode")
            .expression_attribute_values(":user_id", AttributeValue::S(user_id.to_string()))
            .expression_attribute_values(":image_hash", AttributeValue::S(image_hash.to_string()))
            .expression_attribute_values(":processing_mode", AttributeValue::S(mode_str))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        if let Some(items) = result.items {
            if let Some(item) = items.into_iter().next() {
                return Ok(Some(self.parse_inference_cache_from_item(item)?));
            }
        }
        Ok(None)
    }

    pub async fn get_inference_cache_by_image_hash(
        &self,
        user_id: Uuid,
        image_hash: &str,
    ) -> Result<Vec<InferenceCacheEntry>, RepositoryError> {
        let result = self
            .client
            .scan()
            .table_name(&self.inference_table)
            .filter_expression("user_id = :user_id AND image_hash = :image_hash")
            .expression_attribute_values(":user_id", AttributeValue::S(user_id.to_string()))
            .expression_attribute_values(":image_hash", AttributeValue::S(image_hash.to_string()))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        let mut entries = Vec::new();
        if let Some(items) = result.items {
            for item in items {
                entries.push(self.parse_inference_cache_from_item(item)?);
            }
        }
        Ok(entries)
    }

    pub async fn delete_inference_cache(&self, entry_id: Uuid) -> Result<(), RepositoryError> {
        let mut key = HashMap::new();
        key.insert("id".to_string(), AttributeValue::S(entry_id.to_string()));

        self.client
            .delete_item()
            .table_name(&self.inference_table)
            .set_key(Some(key))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        Ok(())
    }

    pub async fn delete_inference_cache_by_image_hash(
        &self,
        user_id: Uuid,
        image_hash: &str,
    ) -> Result<(), RepositoryError> {
        // Get all inference cache entries for this image
        let entries = self
            .get_inference_cache_by_image_hash(user_id, image_hash)
            .await?;

        // Delete each entry
        for entry in entries {
            self.delete_inference_cache(entry.id).await?;
        }

        Ok(())
    }

    pub async fn delete_user_inference_cache(&self, user_id: Uuid) -> Result<(), RepositoryError> {
        // Get all inference cache entries for the user
        let result = self
            .client
            .scan()
            .table_name(&self.inference_table)
            .filter_expression("user_id = :user_id")
            .expression_attribute_values(":user_id", AttributeValue::S(user_id.to_string()))
            .send()
            .await
            .map_err(|e| RepositoryError::DynamoDb(e.to_string()))?;

        if let Some(items) = result.items {
            for item in items {
                let entry = self.parse_inference_cache_from_item(item)?;
                self.delete_inference_cache(entry.id).await?;
            }
        }

        Ok(())
    }

    // Helper methods for parsing DynamoDB items
    fn parse_user_from_item(
        &self,
        item: HashMap<String, AttributeValue>,
    ) -> Result<User, RepositoryError> {
        let id = item
            .get("id")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid user id".to_string()))?;

        // Cognito fields
        let cognito_sub = item
            .get("cognito_sub")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid cognito_sub".to_string()))?
            .clone();

        let cognito_access_token = item
            .get("cognito_access_token")
            .and_then(|v| v.as_s().ok())
            .cloned();

        let cognito_refresh_token = item
            .get("cognito_refresh_token")
            .and_then(|v| v.as_s().ok())
            .cloned();

        // Common fields
        let email = item
            .get("email")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid email".to_string()))?
            .clone();

        let name = item
            .get("name")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid name".to_string()))?
            .clone();

        let picture_url = item.get("picture_url").and_then(|v| v.as_s().ok()).cloned();

        let created_at = item
            .get("created_at")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .ok_or_else(|| RepositoryError::InvalidData("Invalid created_at".to_string()))?;

        let updated_at = item
            .get("updated_at")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or(created_at);

        let last_login = item
            .get("last_login")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let is_active = *item
            .get("is_active")
            .and_then(|v| v.as_bool().ok())
            .unwrap_or(&true);

        Ok(User {
            id,
            cognito_sub,
            cognito_access_token,
            cognito_refresh_token,
            email,
            name,
            picture_url,
            created_at,
            updated_at,
            last_login,
            is_active,
        })
    }

    fn parse_image_cache_from_item(
        &self,
        item: HashMap<String, AttributeValue>,
    ) -> Result<ImageCacheEntry, RepositoryError> {
        let id = item
            .get("id")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid id".to_string()))?;

        let user_id = item
            .get("user_id")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid user_id".to_string()))?;

        let image_hash = item
            .get("image_hash")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid image_hash".to_string()))?
            .clone();

        let file_name = item
            .get("file_name")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid file_name".to_string()))?
            .clone();

        let file_size = item
            .get("file_size")
            .and_then(|v| v.as_n().ok())
            .and_then(|s| s.parse::<i64>().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid file_size".to_string()))?;

        let mime_type = item
            .get("mime_type")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid mime_type".to_string()))?
            .clone();

        let s3_key = item
            .get("s3_key")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid s3_key".to_string()))?
            .clone();

        let s3_bucket = item
            .get("s3_bucket")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid s3_bucket".to_string()))?
            .clone();

        let created_at = item
            .get("created_at")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .ok_or_else(|| RepositoryError::InvalidData("Invalid created_at".to_string()))?;

        let last_accessed = item
            .get("last_accessed")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .ok_or_else(|| RepositoryError::InvalidData("Invalid last_accessed".to_string()))?;

        let access_count = item
            .get("access_count")
            .and_then(|v| v.as_n().ok())
            .and_then(|s| s.parse::<i32>().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid access_count".to_string()))?;

        Ok(ImageCacheEntry {
            id,
            user_id,
            image_hash,
            file_name,
            file_size,
            mime_type,
            s3_key,
            s3_bucket,
            created_at,
            last_accessed,
            access_count,
        })
    }

    fn parse_inference_cache_from_item(
        &self,
        item: HashMap<String, AttributeValue>,
    ) -> Result<InferenceCacheEntry, RepositoryError> {
        let id = item
            .get("id")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid id".to_string()))?;

        let user_id = item
            .get("user_id")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| Uuid::parse_str(s).ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid user_id".to_string()))?;

        let image_hash = item
            .get("image_hash")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid image_hash".to_string()))?
            .clone();

        let processing_mode = item
            .get("processing_mode")
            .and_then(|v| v.as_s().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid processing_mode".to_string()))?
            .clone();

        let predictions = item
            .get("predictions")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| serde_json::from_str(s).ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid predictions".to_string()))?;

        let is_ai = *item
            .get("is_ai")
            .and_then(|v| v.as_bool().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid is_ai".to_string()))?;

        let confidence = item
            .get("confidence")
            .and_then(|v| v.as_n().ok())
            .and_then(|s| s.parse::<f32>().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid confidence".to_string()))?;

        let created_at = item
            .get("created_at")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .ok_or_else(|| RepositoryError::InvalidData("Invalid created_at".to_string()))?;

        let last_accessed = item
            .get("last_accessed")
            .and_then(|v| v.as_s().ok())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .ok_or_else(|| RepositoryError::InvalidData("Invalid last_accessed".to_string()))?;

        let access_count = item
            .get("access_count")
            .and_then(|v| v.as_n().ok())
            .and_then(|s| s.parse::<i32>().ok())
            .ok_or_else(|| RepositoryError::InvalidData("Invalid access_count".to_string()))?;

        Ok(InferenceCacheEntry {
            id,
            user_id,
            image_hash,
            processing_mode,
            predictions,
            is_ai,
            confidence,
            created_at,
            last_accessed,
            access_count,
        })
    }
}
