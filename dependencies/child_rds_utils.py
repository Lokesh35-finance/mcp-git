from child_rds_utils import stop_rds_cluster, check_rds_status
from child_dms_utils import check_dms_task_status

def orchestrate_rds(cluster_id, dms_task_arn):
    print("🔴 Stopping RDS Cluster...")
    stop_rds_cluster(cluster_id)
    print("⏳ Checking RDS status...")
    status = check_rds_status(cluster_id)
    print(f"✅ RDS Status: {status}")

    print("📊 Checking DMS Task status (cross-check dependency)...")
    dms_status = check_dms_task_status(dms_task_arn)
    print(f"ℹ️ DMS Task Status: {dms_status}")
