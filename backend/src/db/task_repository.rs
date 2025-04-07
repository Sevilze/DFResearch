use sqlx::PgPool;
use uuid::Uuid;
use shared::Task;

#[derive(Clone)]
pub struct TaskRepository {
    pool: PgPool,
}

impl TaskRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub async fn create_task_with_id(
        &self,
        id: Uuid,
        user_id: Option<Uuid>,
        image_url: Option<String>,
    ) -> Result<Task, sqlx::Error> {
        let rec = sqlx::query_as!(
            Task,
            r#"
            INSERT INTO tasks (id, user_id, status, created_at, updated_at, image_url)
            VALUES ($1, $2, 'pending', now(), now(), $3)
            RETURNING id, user_id, status, created_at, updated_at, result, error, image_url
            "#,
            id,
            user_id,
            image_url
        )
        .fetch_one(&self.pool)
        .await?;
        Ok(rec)
    }

    pub async fn get_task(&self, id: Uuid) -> Result<Option<Task>, sqlx::Error> {
        let rec = sqlx::query_as!(
            Task,
            r#"
            SELECT id, user_id, status, created_at, updated_at, result, error, image_url
            FROM tasks WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(rec)
    }

    pub async fn update_task_status(
        &self,
        id: Uuid,
        status: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query!(
            r#"
            UPDATE tasks SET status = $1, updated_at = now()
            WHERE id = $2
            "#,
            status,
            id
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn update_task_result(
        &self,
        id: Uuid,
        result: serde_json::Value,
    ) -> Result<(), sqlx::Error> {
        sqlx::query!(
            r#"
            UPDATE tasks SET result = $1, status = 'completed', updated_at = now()
            WHERE id = $2
            "#,
            result,
            id
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn update_task_error(
        &self,
        id: Uuid,
        error_msg: &str,
    ) -> Result<(), sqlx::Error> {
        sqlx::query!(
            r#"
            UPDATE tasks SET error = $1, status = 'failed', updated_at = now()
            WHERE id = $2
            "#,
            error_msg,
            id
        )
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}
