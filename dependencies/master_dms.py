from child_dms_utils import stop_dms_task, check_dms_task_status
from child_rds_utils import check_rds_status

def orchestrate_dms(task_arn, cluster_id):
    print("🔴 Stopping DMS Task...")
    stop_dms_task(task_arn)
    print("⏳ Checking DMS Task status...")
    status = check_dms_task_status(task_arn)
    print(f"✅ DMS Task Status: {status}")

    print("📊 Checking RDS Cluster status (cross-check dependency)...")
    rds_status = check_rds_status(cluster_id)
    print(f"ℹ️ RDS Cluster Status: {rds_status}")
