use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client;
use hex;
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Clone)]
pub struct S3Service {
    client: Client,
    bucket_name: String,
}

#[derive(Debug, thiserror::Error)]
pub enum S3ServiceError {
    #[error("S3 error: {0}")]
    S3(String),
    #[error("Invalid file format")]
    InvalidFormat,
    #[error("File too large")]
    FileTooLarge,
}

impl S3Service {
    pub fn new(client: Client, bucket_name: String) -> Self {
        Self {
            client,
            bucket_name,
        }
    }

    pub fn calculate_image_hash(image_data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(image_data);
        hex::encode(hasher.finalize())
    }

    pub fn generate_s3_key(user_id: Uuid, image_hash: &str, file_extension: &str) -> String {
        format!("images/{}/{}.{}", user_id, image_hash, file_extension)
    }

    pub fn extract_file_extension(mime_type: &str) -> Result<&str, S3ServiceError> {
        match mime_type {
            "image/jpeg" => Ok("jpg"),
            "image/png" => Ok("png"),
            "image/webp" => Ok("webp"),
            "image/gif" => Ok("gif"),
            _ => Err(S3ServiceError::InvalidFormat),
        }
    }

    pub fn validate_image_size(image_data: &[u8]) -> Result<(), S3ServiceError> {
        const MAX_SIZE: usize = 50 * 1024 * 1024;
        if image_data.len() > MAX_SIZE {
            return Err(S3ServiceError::FileTooLarge);
        }
        Ok(())
    }

    pub async fn upload_image(
        &self,
        image_data: &[u8],
        s3_key: &str,
        mime_type: &str,
    ) -> Result<(), S3ServiceError> {
        S3Service::validate_image_size(image_data)?;

        let body = ByteStream::from(image_data.to_vec());

        self.client
            .put_object()
            .bucket(&self.bucket_name)
            .key(s3_key)
            .body(body)
            .content_type(mime_type)
            .send()
            .await
            .map_err(|e| S3ServiceError::S3(e.to_string()))?;

        Ok(())
    }

    pub async fn get_image(&self, s3_key: &str) -> Result<Vec<u8>, S3ServiceError> {
        let result = self
            .client
            .get_object()
            .bucket(&self.bucket_name)
            .key(s3_key)
            .send()
            .await
            .map_err(|e| S3ServiceError::S3(e.to_string()))?;

        let body = result
            .body
            .collect()
            .await
            .map_err(|e| S3ServiceError::S3(e.to_string()))?;
        Ok(body.into_bytes().to_vec())
    }

    pub async fn delete_image(&self, s3_key: &str) -> Result<(), S3ServiceError> {
        self.client
            .delete_object()
            .bucket(&self.bucket_name)
            .key(s3_key)
            .send()
            .await
            .map_err(|e| S3ServiceError::S3(e.to_string()))?;

        Ok(())
    }

    pub async fn delete_images(&self, s3_keys: &[String]) -> Result<(), S3ServiceError> {
        const BATCH_SIZE: usize = 15;

        for chunk in s3_keys.chunks(BATCH_SIZE) {
            let mut delete_futures = Vec::new();

            for s3_key in chunk {
                delete_futures.push(self.delete_image(s3_key));
            }

            for future in delete_futures {
                future.await?;
            }
        }

        Ok(())
    }
}
