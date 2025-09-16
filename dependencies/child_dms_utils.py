import boto3
from child_rds_utils import check_rds_status   # 🔗 new cross-dependency

def stop_dms_task(task_arn):
    client = boto3.client("dms")
    response = client.stop_replication_task(ReplicationTaskArn=task_arn)
    return response

def check_dms_task_status(task_arn):
    client = boto3.client("dms")
    response = client.describe_replication_tasks(
        Filters=[{"Name": "replication-task-arn", "Values": [task_arn]}]
    )
    return response["ReplicationTasks"][0]["Status"]

def ensure_rds_stopped(cluster_id):
    """
    Cross-dependency check:
    Before finalizing DMS stop, verify RDS cluster is stopped.
    """
    status = check_rds_status(cluster_id)
    if status.lower() == "stopped":
        return True
    return False
