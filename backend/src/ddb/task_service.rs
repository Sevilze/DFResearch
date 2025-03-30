use aws_config::meta::region::RegionProviderChain;
use aws_sdk_dynamodb::{ Client, Error as AwsError };
use aws_sdk_dynamodb::types::AttributeValue;
use std::collections::HashMap;
use crate::ddb::model::{ Task, TaskStatus };
use thiserror::Error;
use serde_json;
use log::{ info, debug, error, warn };

#[derive(Error, Debug)]
pub enum TaskServiceError {
    #[error("Task not found: {0}")] NotFound(String),
    #[error("AWS SDK error: {0}")] SdkError(AwsError),
    #[error("IO error: {0}")] IoError(#[from] std::io::Error),
    #[error("Serde JSON error: {0}")] SerdeError(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct TaskService {
    client: Client,
    table_name: String,
}

impl TaskService {
    pub async fn new(table_name: String) -> Result<Self, TaskServiceError> {
        info!("Initializing DynamoDB task service with table: {}", table_name);
        let region_provider = RegionProviderChain::default_provider().or_else("us-east-1");
        let config = aws_config::from_env().region(region_provider).load().await;
        let client = Client::new(&config);
        info!("DynamoDB task service initialized");
        Ok(Self { client, table_name })
    }

    pub async fn create_task(&self) -> Result<Task, TaskServiceError> {
        info!("Creating new task in table: {}", self.table_name);
        let task = Task::new();
        debug!("New task created locally with id: {}", task.id);
        let attributes = self.task_to_attributes(&task)?;
        self.client
            .put_item()
            .table_name(&self.table_name)
            .set_item(Some(attributes))
            .send().await
            .map_err(|e| {
                error!("AWS SDK error during put_item for task {}: {:?}", task.id, e);
                TaskServiceError::SdkError(e.into())
            })?;
        info!("Task {} successfully created in DynamoDB", task.id);
        Ok(task)
    }

    pub async fn get_task(&self, task_id: &str) -> Result<Option<Task>, TaskServiceError> {
        info!("Retrieving task with id: {}", task_id);
        let response = self.client
            .get_item()
            .table_name(&self.table_name)
            .key("id", AttributeValue::S(task_id.to_string()))
            .send().await
            .map_err(|e| {
                error!("AWS SDK error during get_item for task {}: {:?}", task_id, e);
                TaskServiceError::SdkError(e.into())
            })?;
        if let Some(item) = response.item {
            let task = self.attributes_to_task(item)?;
            info!("Task {} retrieved successfully", task_id);
            Ok(Some(task))
        } else {
            warn!("Task {} not found in DynamoDB", task_id);
            Ok(None)
        }
    }

    pub async fn update_task(&self, task: &Task) -> Result<(), TaskServiceError> {
        info!("Updating task {} in DynamoDB", task.id);
        let attributes = self.task_to_attributes(task)?;
        self.client
            .put_item()
            .table_name(&self.table_name)
            .set_item(Some(attributes))
            .send().await
            .map_err(|e| {
                error!("AWS SDK error during update for task {}: {:?}", task.id, e);
                TaskServiceError::SdkError(e.into())
            })?;
        info!("Task {} updated successfully", task.id);
        Ok(())
    }

    pub async fn update_task_status(
        &self,
        task_id: &str,
        status: TaskStatus
    ) -> Result<Task, TaskServiceError> {
        info!("Updating status for task {} to {:?}", task_id, status);
        if let Some(mut task) = self.get_task(task_id).await? {
            task.set_status(status);
            self.update_task(&task).await?;
            info!("Task {} status updated successfully", task_id);
            Ok(task)
        } else {
            error!("Task {} not found for status update", task_id);
            Err(TaskServiceError::NotFound(format!("Task with ID {} not found", task_id)))
        }
    }

    pub async fn update_task_result(
        &self,
        task_id: &str,
        result: Vec<f32>
    ) -> Result<Task, TaskServiceError> {
        info!("Updating result for task {}", task_id);
        if let Some(mut task) = self.get_task(task_id).await? {
            task.set_result(result);
            self.update_task(&task).await?;
            info!("Task {} result updated successfully", task_id);
            Ok(task)
        } else {
            error!("Task {} not found for result update", task_id);
            Err(TaskServiceError::NotFound(format!("Task with ID {} not found", task_id)))
        }
    }

    pub async fn update_task_error(
        &self,
        task_id: &str,
        error: String
    ) -> Result<Task, TaskServiceError> {
        info!("Updating error for task {}", task_id);
        if let Some(mut task) = self.get_task(task_id).await? {
            task.set_error(error);
            self.update_task(&task).await?;
            info!("Task {} error updated successfully", task_id);
            Ok(task)
        } else {
            error!("Task {} not found for error update", task_id);
            Err(TaskServiceError::NotFound(format!("Task with ID {} not found", task_id)))
        }
    }

    fn task_to_attributes(
        &self,
        task: &Task
    ) -> Result<HashMap<String, AttributeValue>, TaskServiceError> {
        debug!("Converting task {} to DynamoDB attributes", task.id);
        let mut attributes = HashMap::new();
        attributes.insert("id".to_string(), AttributeValue::S(task.id.clone()));
        attributes.insert("status".to_string(), AttributeValue::S(format!("{:?}", task.status)));
        attributes.insert("created_at".to_string(), AttributeValue::S(task.created_at.clone()));
        attributes.insert("updated_at".to_string(), AttributeValue::S(task.updated_at.clone()));
        if let Some(result) = &task.result {
            let json = serde_json::to_string(result)?;
            attributes.insert("result".to_string(), AttributeValue::S(json));
        }
        if let Some(error) = &task.error {
            attributes.insert("error".to_string(), AttributeValue::S(error.clone()));
        }
        debug!("Task {} attributes converted successfully", task.id);
        Ok(attributes)
    }

    fn attributes_to_task(
        &self,
        attributes: HashMap<String, AttributeValue>
    ) -> Result<Task, TaskServiceError> {
        debug!("Converting DynamoDB attributes into task");
        let id = attributes
            .get("id")
            .and_then(|av| av.as_s().ok())
            .ok_or_else(|| TaskServiceError::NotFound("Missing id attribute".to_string()))?
            .to_string();
        let status_str = attributes
            .get("status")
            .and_then(|av| av.as_s().ok())
            .ok_or_else(|| TaskServiceError::NotFound("Missing status attribute".to_string()))?
            .to_string();
        let status = match status_str.as_str() {
            "\"PENDING\"" | "Pending" => TaskStatus::Pending,
            "\"PROCESSING\"" | "Processing" => TaskStatus::Processing,
            "\"COMPLETED\"" | "Completed" => TaskStatus::Completed,
            "\"FAILED\"" | "Failed" => TaskStatus::Failed,
            _ => TaskStatus::Pending,
        };
        let created_at = attributes
            .get("created_at")
            .and_then(|av| av.as_s().ok())
            .ok_or_else(|| TaskServiceError::NotFound("Missing created_at attribute".to_string()))?
            .to_string();
        let updated_at = attributes
            .get("updated_at")
            .and_then(|av| av.as_s().ok())
            .ok_or_else(|| TaskServiceError::NotFound("Missing updated_at attribute".to_string()))?
            .to_string();
        let result = if
            let Some(result_av) = attributes.get("result").and_then(|av| av.as_s().ok())
        {
            let result_vec: Vec<f32> = serde_json::from_str(result_av)?;
            Some(result_vec)
        } else {
            None
        };
        let error = attributes
            .get("error")
            .and_then(|av| av.as_s().ok())
            .map(|s| s.to_string());
        debug!("DynamoDB attributes converted into task {}", id);
        Ok(Task {
            id,
            status,
            created_at,
            updated_at,
            result,
            error,
        })
    }
}
