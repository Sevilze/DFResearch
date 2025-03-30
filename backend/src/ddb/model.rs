use serde::{Deserialize, Serialize};
use chrono::Utc;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TaskStatus {
    #[serde(rename = "PENDING")]
    Pending,
    #[serde(rename = "PROCESSING")]
    Processing,
    #[serde(rename = "COMPLETED")]
    Completed,
    #[serde(rename = "FAILED")]
    Failed,
}

impl Default for TaskStatus {
    fn default() -> Self {
        TaskStatus::Pending
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Task {
    pub id: String,
    pub status: TaskStatus,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl Task {
    pub fn new() -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            id: Uuid::new_v4().to_string(),
            status: TaskStatus::default(),
            created_at: now.clone(),
            updated_at: now,
            result: None,
            error: None,
        }
    }

    pub fn set_status(&mut self, status: TaskStatus) {
        self.status = status;
        self.updated_at = Utc::now().to_rfc3339();
    }

    pub fn set_result(&mut self, result: Vec<f32>) {
        self.result = Some(result);
        self.status = TaskStatus::Completed;
        self.updated_at = Utc::now().to_rfc3339();
    }

    pub fn set_error(&mut self, error: String) {
        self.error = Some(error);
        self.status = TaskStatus::Failed;
        self.updated_at = Utc::now().to_rfc3339();
    }
}