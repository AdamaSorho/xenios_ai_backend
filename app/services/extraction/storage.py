"""S3/R2 storage service for document extraction."""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import aioboto3
from botocore.config import Config

from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    """
    Handle file uploads and downloads to S3-compatible storage.

    Supports both AWS S3 and Cloudflare R2 (S3-compatible).
    """

    def __init__(self) -> None:
        """Initialize the storage service."""
        self._session: aioboto3.Session | None = None
        self._settings = get_settings()

    @property
    def session(self) -> aioboto3.Session:
        """Get or create the aioboto3 session."""
        if self._session is None:
            self._session = aioboto3.Session()
        return self._session

    def _get_client_config(self) -> dict:
        """Get the S3 client configuration."""
        config: dict = {
            "service_name": "s3",
            "aws_access_key_id": self._settings.s3_access_key_id,
            "aws_secret_access_key": self._settings.s3_secret_access_key,
            "config": Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        }

        # Add endpoint URL for R2 or other S3-compatible services
        if self._settings.s3_endpoint_url:
            config["endpoint_url"] = self._settings.s3_endpoint_url

        # Add region if specified
        if self._settings.s3_region and self._settings.s3_region != "auto":
            config["region_name"] = self._settings.s3_region

        return config

    def _generate_storage_key(
        self,
        client_id: str,
        filename: str,
        prefix: str = "extractions",
    ) -> str:
        """
        Generate a unique storage key for a file.

        Format: {prefix}/{client_id}/{year}/{month}/{uuid}_{filename}
        """
        now = datetime.utcnow()
        unique_id = uuid4().hex[:8]
        safe_filename = Path(filename).name  # Remove any directory components
        return f"{prefix}/{client_id}/{now.year}/{now.month:02d}/{unique_id}_{safe_filename}"

    async def upload_file(
        self,
        file_content: bytes | BinaryIO,
        client_id: str,
        filename: str,
        content_type: str,
    ) -> str:
        """
        Upload a file to S3/R2 storage.

        Args:
            file_content: File content as bytes or file-like object
            client_id: Client ID for organizing storage
            filename: Original filename
            content_type: MIME type of the file

        Returns:
            S3 URL of the uploaded file (s3://bucket/key)
        """
        key = self._generate_storage_key(client_id, filename)

        logger.info(
            "Uploading file to storage",
            bucket=self._settings.s3_bucket,
            key=key,
            content_type=content_type,
        )

        async with self.session.client(**self._get_client_config()) as s3:
            # Handle both bytes and file-like objects
            if isinstance(file_content, bytes):
                await s3.put_object(
                    Bucket=self._settings.s3_bucket,
                    Key=key,
                    Body=file_content,
                    ContentType=content_type,
                )
            else:
                await s3.upload_fileobj(
                    file_content,
                    self._settings.s3_bucket,
                    key,
                    ExtraArgs={"ContentType": content_type},
                )

        logger.info("File uploaded successfully", key=key)
        return f"s3://{self._settings.s3_bucket}/{key}"

    async def download_file(self, file_url: str) -> bytes:
        """
        Download a file from S3/R2 storage.

        Args:
            file_url: S3 URL (s3://bucket/key) or just the key

        Returns:
            File content as bytes
        """
        bucket, key = self._parse_s3_url(file_url)

        logger.info("Downloading file from storage", bucket=bucket, key=key)

        async with self.session.client(**self._get_client_config()) as s3:
            response = await s3.get_object(Bucket=bucket, Key=key)
            content = await response["Body"].read()

        logger.info("File downloaded successfully", key=key, size=len(content))
        return content

    async def download_to_temp_file(self, file_url: str) -> str:
        """
        Download a file to a temporary local file.

        Args:
            file_url: S3 URL of the file

        Returns:
            Path to the temporary file (caller must delete)
        """
        content = await self.download_file(file_url)

        # Preserve the original extension
        _, key = self._parse_s3_url(file_url)
        ext = Path(key).suffix

        # Create temp file with proper extension
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        try:
            os.write(fd, content)
        finally:
            os.close(fd)

        logger.debug("File downloaded to temp", temp_path=temp_path)
        return temp_path

    async def get_presigned_url(
        self,
        file_url: str,
        expires: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for temporary access.

        Args:
            file_url: S3 URL of the file
            expires: Expiration time in seconds (default 1 hour)

        Returns:
            Presigned URL for direct access
        """
        bucket, key = self._parse_s3_url(file_url)

        async with self.session.client(**self._get_client_config()) as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires,
            )

        return url

    async def delete_file(self, file_url: str) -> None:
        """
        Delete a file from S3/R2 storage.

        Args:
            file_url: S3 URL of the file to delete
        """
        bucket, key = self._parse_s3_url(file_url)

        logger.info("Deleting file from storage", bucket=bucket, key=key)

        async with self.session.client(**self._get_client_config()) as s3:
            await s3.delete_object(Bucket=bucket, Key=key)

        logger.info("File deleted successfully", key=key)

    async def file_exists(self, file_url: str) -> bool:
        """
        Check if a file exists in storage.

        Args:
            file_url: S3 URL of the file

        Returns:
            True if file exists, False otherwise
        """
        bucket, key = self._parse_s3_url(file_url)

        try:
            async with self.session.client(**self._get_client_config()) as s3:
                await s3.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False

    def _parse_s3_url(self, url: str) -> tuple[str, str]:
        """
        Parse an S3 URL into bucket and key.

        Args:
            url: S3 URL (s3://bucket/key) or just the key

        Returns:
            Tuple of (bucket, key)
        """
        if url.startswith("s3://"):
            # Parse s3://bucket/key format
            parts = url[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            # Assume it's just a key
            bucket = self._settings.s3_bucket
            key = url

        return bucket, key


# Global storage service instance
_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    """Get the global storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
