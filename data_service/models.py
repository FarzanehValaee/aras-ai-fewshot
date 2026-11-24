import secrets

from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import Q
from django.utils import timezone


def default_dict():
    return {}


def default_list():
    return []


class ApiKey(models.Model):
    name = models.CharField(max_length=255, unique=True)
    token = models.CharField(max_length=128, unique=True, db_index=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    def save(self, *args, **kwargs):
        if not self.token:
            self.token = f"aras-ai-data-{secrets.token_urlsafe(32)}"
        super().save(*args, **kwargs)


class DataSourceType(models.TextChoices):
    UPLOAD = "upload", "Upload"
    LINK = "link", "Link"


class DatabaseEngine(models.TextChoices):
    MYSQL = "mysql", "MySQL"
    SQLSERVER = "sqlserver", "SQL Server"
    POSTGRESQL = "postgresql", "PostgreSQL"


class ConnectionStatus(models.TextChoices):
    CONNECTED = "connected", "Connected"
    FAILED = "failed", "Failed"
    CHECKING = "checking", "Checking"
    UNKNOWN = "unknown", "Unknown"


class DataSource(models.Model):
    """Reusable analytics data source (upload or linked DB)."""

    tenant = models.CharField(max_length=64, blank=True, default="", db_index=True)
    owner_external_id = models.CharField(max_length=128, db_index=True)
    type = models.CharField(max_length=16, choices=DataSourceType.choices, db_index=True)
    last_status = models.CharField(
        max_length=16,
        choices=ConnectionStatus.choices,
        default=ConnectionStatus.UNKNOWN,
        db_index=True,
    )
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    schema_description = models.TextField(blank=True, default="")
    original_file = models.FileField(
        upload_to="uploads/%Y/%m/",
        blank=True,
        null=True,
        max_length=500,
    )
    address_port = models.CharField(max_length=255, blank=True, default="")
    username = models.CharField(max_length=255, blank=True, default="")
    password = models.CharField(max_length=255, blank=True, default="")
    db_name = models.CharField(max_length=255, blank=True, default="")
    db_engine = models.CharField(
        max_length=16,
        choices=DatabaseEngine.choices,
        blank=True,
        default=DatabaseEngine.MYSQL,
    )
    db_schema = models.CharField(max_length=64, blank=True, null=True, db_index=True)
    status_message = models.TextField(blank=True, null=True)
    status_details = models.JSONField(blank=True, default=default_dict)
    last_error = models.TextField(blank=True, null=True)
    last_error_at = models.DateTimeField(blank=True, null=True)
    meta = models.JSONField(blank=True, default=default_dict)
    column_catalog = models.JSONField(blank=True, default=default_dict)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["tenant", "owner_external_id", "created_at"]),
            models.Index(fields=["type"]),
            models.Index(fields=["last_status"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "owner_external_id", "title"],
                name="uniq_tenant_owner_title",
            ),
            models.UniqueConstraint(
                fields=["tenant", "owner_external_id", "db_schema"],
                condition=Q(db_schema__isnull=False) & ~Q(db_schema=""),
                name="uniq_tenant_owner_db_schema",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.title} ({self.tenant}/{self.owner_external_id})"

    def clean(self):
        super().clean()
        if self.type == DataSourceType.UPLOAD and not self.original_file:
            raise ValidationError(
                {"original_file": "Archive file is required for upload type."}
            )
        if self.type == DataSourceType.LINK:
            if not self.db_name:
                raise ValidationError({"db_name": "Database name is required for link type."})
            if not self.address_port:
                raise ValidationError(
                    {"address_port": "Address and port are required for link type."}
                )

    @property
    def host(self) -> str:
        raw = (self.address_port or "").strip()
        if not raw:
            return ""
        # maxsplit=1 so IPv4 hosts like 127.0.0.1:3306 stay intact
        return raw.split(":", 1)[0].strip()

    @property
    def port(self) -> int | None:
        raw = (self.address_port or "").strip()
        if ":" not in raw:
            return None
        try:
            return int(raw.split(":", 1)[1].strip())
        except Exception:
            return None

    def mark_error(self, message: str) -> None:
        self.last_status = ConnectionStatus.FAILED
        self.last_error = message
        self.last_error_at = timezone.now()
        self.status_message = message
        self.save(
            update_fields=[
                "last_status",
                "last_error",
                "last_error_at",
                "status_message",
                "updated_at",
            ]
        )


class UploadJobStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    RUNNING = "running", "Running"
    DONE = "done", "Done"
    FAILED = "failed", "Failed"


class UploadJob(models.Model):
    data_source = models.ForeignKey(
        DataSource, on_delete=models.CASCADE, related_name="upload_jobs"
    )
    status = models.CharField(
        max_length=16,
        choices=UploadJobStatus.choices,
        default=UploadJobStatus.PENDING,
        db_index=True,
    )
    attempt = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True, default="")
    created_tables = models.JSONField(blank=True, default=default_list)
    failed = models.JSONField(blank=True, default=default_list)
    skipped = models.JSONField(blank=True, default=default_list)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    started_at = models.DateTimeField(blank=True, null=True)
    finished_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"UploadJob#{self.pk} source={self.data_source_id} ({self.status})"

    def start(self):
        self.status = UploadJobStatus.RUNNING
        self.attempt = int(self.attempt or 0) + 1
        self.started_at = timezone.now()
        self.error_message = ""
        self.save(
            update_fields=["status", "attempt", "started_at", "error_message"]
        )

    def succeed(self, *, created_tables=None, failed=None, skipped=None):
        self.status = UploadJobStatus.DONE
        self.created_tables = created_tables or []
        self.failed = failed or []
        self.skipped = skipped or []
        self.finished_at = timezone.now()
        self.error_message = ""
        self.save(
            update_fields=[
                "status",
                "created_tables",
                "failed",
                "skipped",
                "finished_at",
                "error_message",
            ]
        )

    def fail(self, message: str):
        self.status = UploadJobStatus.FAILED
        self.error_message = message or "Upload failed"
        self.finished_at = timezone.now()
        self.save(update_fields=["status", "error_message", "finished_at"])


class FewShotExample(models.Model):
    data_source = models.ForeignKey(
        DataSource, on_delete=models.CASCADE, related_name="fewshot_examples"
    )
    title = models.CharField(max_length=255, blank=True, default="")
    user_question = models.TextField()
    ideal_answer = models.TextField(blank=True, default="")
    ideal_sql = models.TextField(blank=True, default="")
    intent = models.CharField(max_length=64, blank=True, default="")
    is_active = models.BooleanField(default=True)
    priority = models.IntegerField(default=100)
    generation_params = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["priority", "-created_at"]
