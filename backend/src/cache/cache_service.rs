use shared::{InferenceResponse, ProcessingMode};
use uuid::Uuid;

use crate::cache::models::{CacheHistoryEntry, ImageCacheEntry, InferenceCacheEntry};
use crate::db::dynamodb_repository::{DynamoDbRepository, RepositoryError};
use crate::storage::s3_service::{S3Service, S3ServiceError};

#[derive(Clone)]
pub struct CacheService {
    db_repo: DynamoDbRepository,
    s3_service: S3Service,
}

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("Repository error: {0}")]
    Repository(#[from] RepositoryError),
    #[error("Storage error: {0}")]
    Storage(#[from] S3ServiceError),
    #[error("Cache miss")]
    CacheMiss,
}

impl CacheService {
    pub fn new(db_repo: DynamoDbRepository, s3_service: S3Service) -> Self {
        Self {
            db_repo,
            s3_service,
        }
    }

    pub async fn cache_image(
        &self,
        user_id: Uuid,
        image_data: &[u8],
        file_name: String,
        mime_type: String,
    ) -> Result<ImageCacheEntry, CacheError> {
        let image_hash = S3Service::calculate_image_hash(image_data);

        // Check if image already exists in cache
        if let Some(existing_entry) = self
            .db_repo
            .get_image_cache_by_hash(user_id, &image_hash)
            .await?
        {
            return Ok(existing_entry);
        }

        let file_extension = S3Service::extract_file_extension(&mime_type)?;
        let s3_key = S3Service::generate_s3_key(user_id, &image_hash, file_extension);

        self.s3_service
            .upload_image(image_data, &s3_key, &mime_type)
            .await?;

        let cache_entry = ImageCacheEntry::new(
            user_id,
            image_hash,
            file_name,
            image_data.len() as i64,
            mime_type,
            s3_key,
            "dfresearch-images".to_string(),
        );

        self.db_repo.create_image_cache(&cache_entry).await?;
        Ok(cache_entry)
    }

    pub async fn cache_inference_result(
        &self,
        user_id: Uuid,
        image_hash: String,
        processing_mode: &ProcessingMode,
        predictions: &[f32],
        is_ai: bool,
        confidence: f32,
    ) -> Result<InferenceCacheEntry, CacheError> {
        let cache_entry = InferenceCacheEntry::from_inference_result(
            user_id,
            image_hash,
            processing_mode,
            predictions,
            is_ai,
            confidence,
        );

        self.db_repo.create_inference_cache(&cache_entry).await?;

        Ok(cache_entry)
    }

    pub async fn get_cached_inference(
        &self,
        user_id: Uuid,
        image_hash: &str,
        processing_mode: &ProcessingMode,
    ) -> Result<InferenceResponse, CacheError> {
        let cache_entry = self
            .db_repo
            .get_inference_cache(user_id, image_hash, processing_mode)
            .await?
            .ok_or(CacheError::CacheMiss)?;

        Ok(cache_entry.to_inference_response())
    }

    pub async fn get_user_cache_history(
        &self,
        user_id: Uuid,
    ) -> Result<Vec<CacheHistoryEntry>, CacheError> {
        // Get all image cache entries for the user
        let image_entries = self.db_repo.get_user_image_cache(user_id).await?;
        let mut cache_history = Vec::new();

        for image_entry in image_entries {
            // Get all inference results for this image
            let inference_entries = self
                .db_repo
                .get_inference_cache_by_image_hash(user_id, &image_entry.image_hash)
                .await?;

            let history_entry = CacheHistoryEntry::new(image_entry, inference_entries);
            cache_history.push(history_entry);
        }

        cache_history.sort_by(|a, b| b.image_created_at.cmp(&a.image_created_at));
        Ok(cache_history)
    }

    pub async fn get_cached_image_metadata(
        &self,
        user_id: Uuid,
        image_hash: &str,
    ) -> Result<ImageCacheEntry, CacheError> {
        self.db_repo
            .get_image_cache_by_hash(user_id, image_hash)
            .await?
            .ok_or(CacheError::CacheMiss)
    }

    pub async fn get_cached_image_data(&self, s3_key: &str) -> Result<Vec<u8>, CacheError> {
        self.s3_service
            .get_image(s3_key)
            .await
            .map_err(CacheError::Storage)
    }

    pub async fn delete_cached_image(
        &self,
        user_id: Uuid,
        image_hash: &str,
    ) -> Result<(), CacheError> {
        // Get image metadata to find S3 key
        let image_entry = self.get_cached_image_metadata(user_id, image_hash).await?;

        // Delete from S3
        self.s3_service.delete_image(&image_entry.s3_key).await?;

        self.db_repo
            .delete_inference_cache_by_image_hash(user_id, image_hash)
            .await?;

        self.db_repo
            .delete_image_cache_by_hash(user_id, image_hash)
            .await?;

        Ok(())
    }

    pub async fn clear_user_cache(&self, user_id: Uuid) -> Result<(), CacheError> {
        // Get all image cache entries to find S3 keys
        let image_entries = self.db_repo.get_user_image_cache(user_id).await?;

        let s3_keys: Vec<String> = image_entries
            .iter()
            .map(|entry| entry.s3_key.clone())
            .collect();

        // Delete all images from S3
        if !s3_keys.is_empty() {
            self.s3_service.delete_images(&s3_keys).await?;
        }

        self.db_repo.delete_user_inference_cache(user_id).await?;
        self.db_repo.delete_user_image_cache(user_id).await?;

        Ok(())
    }
}
